"""
EMDB-1050 Dataset Generation Script

This script downloads the cryo-EM volume from EMDB-1050 (PDB ID 6A5L)
and generates 2D projections with metadata, as described in the paper's
Data Availability Statement.

Citation: The cryo-EM volume is publicly hosted by the Electron Microscopy 
Data Bank (https://www.ebi.ac.uk/emdb/EMD-1050).
"""

import numpy as np
import urllib.request
import gzip
import os
from pathlib import Path
from typing import Tuple, List
import h5py
from tqdm import tqdm

class EMDBProjectionGenerator:
    """Generate 2D projections from EMDB-1050 3D volume"""
    
    def __init__(self, data_dir: str = './data/EMDB-1050'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # EMDB-1050 URLs
        self.emdb_url = 'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1050/map/emd_1050.map.gz'
        self.map_file = self.data_dir / 'emd_1050.map'
        self.projections_file = self.data_dir / 'projections.h5'
        
    def download_volume(self):
        """Download EMDB-1050 volume if not already present"""
        if self.map_file.exists():
            print(f"EMDB-1050 volume already downloaded: {self.map_file}")
            return
        
        print("Downloading EMDB-1050 volume...")
        gz_file = self.data_dir / 'emd_1050.map.gz'
        
        try:
            # Download compressed file
            urllib.request.urlretrieve(self.emdb_url, gz_file)
            print(f"Downloaded to {gz_file}")
            
            # Decompress
            print("Decompressing...")
            with gzip.open(gz_file, 'rb') as f_in:
                with open(self.map_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up compressed file
            gz_file.unlink()
            print(f"Volume saved to {self.map_file}")
            
        except Exception as e:
            print(f"Error downloading EMDB-1050: {e}")
            print("Please download manually from https://www.ebi.ac.uk/emdb/EMD-1050")
            raise
    
    def read_mrc_map(self) -> np.ndarray:
        """
        Read MRC/CCP4 format electron density map.
        
        Returns:
            3D numpy array of electron density
        """
        if not self.map_file.exists():
            raise FileNotFoundError(f"Map file not found: {self.map_file}")
        
        # MRC file format parsing (simplified)
        with open(self.map_file, 'rb') as f:
            # Read header (1024 bytes)
            header = np.frombuffer(f.read(1024), dtype=np.int32)
            
            # Grid dimensions (columns, rows, sections)
            nx, ny, nz = header[0], header[1], header[2]
            
            # Data type mode
            mode = header[3]
            
            # Read density data
            if mode == 2:  # 32-bit float
                dtype = np.float32
            elif mode == 0:  # 8-bit signed int
                dtype = np.int8
            elif mode == 1:  # 16-bit signed int
                dtype = np.int16
            else:
                dtype = np.float32
            
            # Read the 3D volume
            volume = np.frombuffer(f.read(), dtype=dtype)
            volume = volume.reshape((nz, ny, nx))
        
        print(f"Loaded volume shape: {volume.shape}")
        return volume.astype(np.float32)
    
    def generate_projection_angles(
        self, 
        num_projections: int = 6000
    ) -> List[Tuple[float, float, float]]:
        """
        Generate uniformly distributed projection angles (Euler angles).
        
        Args:
            num_projections: Number of projections to generate
        
        Returns:
            List of (phi, theta, psi) Euler angle tuples in radians
        """
        angles = []
        
        # Fibonacci sphere for uniform distribution
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(num_projections):
            # Uniform theta (polar angle)
            theta = np.arccos(1 - 2 * (i + 0.5) / num_projections)
            
            # Golden angle phi (azimuthal angle)
            phi = 2 * np.pi * i / golden_ratio
            
            # Random in-plane rotation psi
            psi = np.random.uniform(0, 2 * np.pi)
            
            angles.append((phi, theta, psi))
        
        return angles
    
    def rotation_matrix(
        self, 
        phi: float, 
        theta: float, 
        psi: float
    ) -> np.ndarray:
        """
        Compute 3D rotation matrix from Euler angles (ZYZ convention).
        
        Args:
            phi, theta, psi: Euler angles in radians
        
        Returns:
            3x3 rotation matrix
        """
        # Rotation around Z by phi
        Rz1 = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi),  np.cos(phi), 0],
            [0,            0,           1]
        ])
        
        # Rotation around Y by theta
        Ry = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # Rotation around Z by psi
        Rz2 = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0,            0,           1]
        ])
        
        return Rz2 @ Ry @ Rz1
    
    def project_volume(
        self, 
        volume: np.ndarray, 
        phi: float, 
        theta: float, 
        psi: float,
        output_size: int = 64
    ) -> np.ndarray:
        """
        Generate a 2D projection of the 3D volume.
        
        Args:
            volume: 3D electron density map
            phi, theta, psi: Euler angles for projection
            output_size: Size of output projection image
        
        Returns:
            2D projection image
        """
        nz, ny, nx = volume.shape
        
        # Get rotation matrix
        R = self.rotation_matrix(phi, theta, psi)
        
        # Center coordinates
        center = np.array([nx / 2, ny / 2, nz / 2])
        
        # Create output grid
        projection = np.zeros((output_size, output_size), dtype=np.float32)
        
        # Simple ray-tracing projection (summation along z-axis after rotation)
        for i in range(output_size):
            for j in range(output_size):
                # Map output pixel to volume coordinates
                x_out = (i - output_size / 2) / output_size * min(nx, ny, nz)
                y_out = (j - output_size / 2) / output_size * min(nx, ny, nz)
                
                # Sum along projection direction
                total = 0.0
                count = 0
                
                for k in range(int(min(nx, ny, nz))):
                    # 3D point in projection coordinates
                    point_proj = np.array([x_out, y_out, 
                                          k - min(nx, ny, nz) / 2])
                    
                    # Rotate back to volume coordinates
                    point_vol = R.T @ point_proj + center
                    
                    # Interpolate volume value (nearest neighbor for simplicity)
                    xi, yi, zi = int(round(point_vol[0])), \
                                 int(round(point_vol[1])), \
                                 int(round(point_vol[2]))
                    
                    if 0 <= xi < nx and 0 <= yi < ny and 0 <= zi < nz:
                        total += volume[zi, yi, xi]
                        count += 1
                
                if count > 0:
                    projection[i, j] = total
        
        # Normalize
        if projection.max() > 0:
            projection = projection / projection.max()
        
        return projection
    
    def generate_dataset(
        self, 
        num_train: int = 5000, 
        num_test: int = 1000,
        output_size: int = 64
    ):
        """
        Generate complete EMDB-1050 projection dataset.
        
        Args:
            num_train: Number of training projections
            num_test: Number of test projections
            output_size: Size of projection images
        """
        # Download and load volume
        self.download_volume()
        volume = self.read_mrc_map()
        
        # Normalize volume
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        # Generate angles
        total_projections = num_train + num_test
        angles = self.generate_projection_angles(total_projections)
        
        # Create HDF5 file for projections
        with h5py.File(self.projections_file, 'w') as hf:
            # Create datasets
            train_images = hf.create_dataset(
                'train/images', 
                shape=(num_train, output_size, output_size),
                dtype=np.float32
            )
            train_angles_ds = hf.create_dataset(
                'train/angles',
                shape=(num_train, 3),
                dtype=np.float32
            )
            
            test_images = hf.create_dataset(
                'test/images',
                shape=(num_test, output_size, output_size),
                dtype=np.float32
            )
            test_angles_ds = hf.create_dataset(
                'test/angles',
                shape=(num_test, 3),
                dtype=np.float32
            )
            
            # Generate training projections
            print(f"Generating {num_train} training projections...")
            for i in tqdm(range(num_train)):
                phi, theta, psi = angles[i]
                projection = self.project_volume(volume, phi, theta, psi, output_size)
                train_images[i] = projection
                train_angles_ds[i] = [phi, theta, psi]
            
            # Generate test projections
            print(f"Generating {num_test} test projections...")
            for i in tqdm(range(num_test)):
                phi, theta, psi = angles[num_train + i]
                projection = self.project_volume(volume, phi, theta, psi, output_size)
                test_images[i] = projection
                test_angles_ds[i] = [phi, theta, psi]
            
            # Store metadata
            hf.attrs['emdb_id'] = 'EMD-1050'
            hf.attrs['pdb_id'] = '6A5L'
            hf.attrs['resolution'] = '3.3'  # Angstroms
            hf.attrs['num_train'] = num_train
            hf.attrs['num_test'] = num_test
            hf.attrs['output_size'] = output_size
        
        print(f"Dataset saved to {self.projections_file}")
        print(f"  Training: {num_train} projections")
        print(f"  Test: {num_test} projections")
        print(f"  Image size: {output_size}x{output_size}")


def main():
    """Main function to generate EMDB-1050 dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate EMDB-1050 projection dataset for phase retrieval'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data/EMDB-1050',
        help='Directory to store data'
    )
    parser.add_argument(
        '--num_train', 
        type=int, 
        default=5000,
        help='Number of training projections'
    )
    parser.add_argument(
        '--num_test', 
        type=int, 
        default=1000,
        help='Number of test projections'
    )
    parser.add_argument(
        '--output_size', 
        type=int, 
        default=64,
        help='Size of projection images'
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = EMDBProjectionGenerator(args.data_dir)
    generator.generate_dataset(
        num_train=args.num_train,
        num_test=args.num_test,
        output_size=args.output_size
    )


if __name__ == '__main__':
    main()

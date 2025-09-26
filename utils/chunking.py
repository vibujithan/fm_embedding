import numpy as np
import torch
from typing import Tuple, Optional, Union
import nibabel as nib
import os
from tqdm import tqdm


def create_overlapping_chunks(
    data: Union[np.ndarray, torch.Tensor],
    chunk_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: Tuple[int, int, int] = (32, 32, 32),
    pad_mode: str = 'constant',
    pad_value: float = 0.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Create overlapping chunks from 4D data (modalities, x, y, z) and append them along modalities dimension.
    
    Args:
        data: Input data of shape (modalities, x, y, z)
        chunk_size: Size of each chunk (x, y, z)
        overlap: Overlap between chunks (x, y, z)
        pad_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
        pad_value: Value for constant padding
        
    Returns:
        Chunked data of shape (modalities * num_chunks, chunk_x, chunk_y, chunk_z)
    """
    if isinstance(data, torch.Tensor):
        is_torch = True
        data = data.detach().cpu().numpy()
    else:
        is_torch = False
    
    modalities, x, y, z = data.shape
    chunk_x, chunk_y, chunk_z = chunk_size
    overlap_x, overlap_y, overlap_z = overlap
    
    # Calculate step sizes (non-overlapping portion)
    step_x = chunk_x - overlap_x
    step_y = chunk_y - overlap_y
    step_z = chunk_z - overlap_z
    
    # Calculate number of chunks in each dimension
    num_chunks_x = max(1, int(np.ceil((x - overlap_x) / step_x)))
    num_chunks_y = max(1, int(np.ceil((y - overlap_y) / step_y)))
    num_chunks_z = max(1, int(np.ceil((z - overlap_z) / step_z)))
    
    total_chunks = num_chunks_x * num_chunks_y * num_chunks_z
    
    print(f"Original data shape: ({modalities}, {x}, {y}, {z})")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Step sizes: ({step_x}, {step_y}, {step_z})")
    print(f"Number of chunks: {num_chunks_x} × {num_chunks_y} × {num_chunks_z} = {total_chunks}")
    print(f"Output shape will be: ({modalities * total_chunks}, {chunk_x}, {chunk_y}, {chunk_z})")
    
    # Pad the data if necessary to ensure all chunks are complete
    pad_x = max(0, (num_chunks_x - 1) * step_x + chunk_x - x)
    pad_y = max(0, (num_chunks_y - 1) * step_y + chunk_y - y)
    pad_z = max(0, (num_chunks_z - 1) * step_z + chunk_z - z)
    
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        print(f"Padding data: ({pad_x}, {pad_y}, {pad_z})")
        data = np.pad(data, ((0, 0), (0, pad_x), (0, pad_y), (0, pad_z)), 
                      mode=pad_mode, constant_values=pad_value)
    
    # Initialize output array
    output_shape = (modalities * total_chunks, chunk_x, chunk_y, chunk_z)
    chunked_data = np.zeros(output_shape, dtype=data.dtype)
    
    chunk_idx = 0
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            for k in range(num_chunks_z):
                # Calculate chunk boundaries
                start_x = i * step_x
                end_x = start_x + chunk_x
                start_y = j * step_y
                end_y = start_y + chunk_y
                start_z = k * step_z
                end_z = start_z + chunk_z
                
                # Extract chunk for all modalities
                chunk = data[:, start_x:end_x, start_y:end_y, start_z:end_z]
                
                # Place in output array
                start_mod = chunk_idx * modalities
                end_mod = start_mod + modalities
                chunked_data[start_mod:end_mod] = chunk
                
                chunk_idx += 1
    
    if is_torch:
        chunked_data = torch.from_numpy(chunked_data)
    
    return chunked_data


def reconstruct_from_chunks(
    chunked_data: Union[np.ndarray, torch.Tensor],
    original_shape: Tuple[int, int, int, int],
    chunk_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: Tuple[int, int, int] = (32, 32, 32),
    reconstruction_method: str = 'average'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reconstruct original data from chunked data using specified reconstruction method.
    
    Args:
        chunked_data: Chunked data of shape (modalities * num_chunks, chunk_x, chunk_y, chunk_z)
        original_shape: Original data shape (modalities, x, y, z)
        chunk_size: Size of each chunk (x, y, z)
        overlap: Overlap between chunks (x, y, z)
        reconstruction_method: Method for handling overlapping regions ('average', 'max', 'min')
        
    Returns:
        Reconstructed data of original shape
    """
    if isinstance(chunked_data, torch.Tensor):
        is_torch = True
        chunked_data = chunked_data.detach().cpu().numpy()
    else:
        is_torch = False
    
    modalities, x, y, z = original_shape
    chunk_x, chunk_y, chunk_z = chunk_size
    overlap_x, overlap_y, overlap_z = overlap
    
    # Calculate step sizes
    step_x = chunk_x - overlap_x
    step_y = chunk_y - overlap_y
    step_z = chunk_z - overlap_z
    
    # Calculate number of chunks
    num_chunks_x = max(1, int(np.ceil((x - overlap_x) / step_x)))
    num_chunks_y = max(1, int(np.ceil((y - overlap_y) / step_y)))
    num_chunks_z = max(1, int(np.ceil((z - overlap_z) / step_z)))
    
    total_chunks = num_chunks_x * num_chunks_y * num_chunks_z
    
    # Initialize reconstruction arrays
    reconstructed = np.zeros(original_shape, dtype=chunked_data.dtype)
    count_map = np.zeros(original_shape, dtype=np.float32)
    
    chunk_idx = 0
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            for k in range(num_chunks_z):
                # Calculate chunk boundaries
                start_x = i * step_x
                end_x = min(start_x + chunk_x, x)
                start_y = j * step_y
                end_y = min(start_y + chunk_y, y)
                start_z = k * step_z
                end_z = min(start_z + chunk_z, z)
                
                # Extract chunk from chunked data
                start_mod = chunk_idx * modalities
                end_mod = start_mod + modalities
                chunk = chunked_data[start_mod:end_mod, :end_x-start_x, :end_y-start_y, :end_z-start_z]
                
                # Add to reconstruction
                reconstructed[:, start_x:end_x, start_y:end_y, start_z:end_z] += chunk
                count_map[:, start_x:end_x, start_y:end_y, start_z:end_z] += 1
                
                chunk_idx += 1
    
    # Normalize by count (handles overlapping regions)
    count_map[count_map == 0] = 1  # Avoid division by zero
    reconstructed = reconstructed / count_map
    
    if is_torch:
        reconstructed = torch.from_numpy(reconstructed)
    
    return reconstructed


def process_npy_file(
    input_path: str,
    output_path: str,
    chunk_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: Tuple[int, int, int] = (32, 32, 32),
    pad_mode: str = 'constant',
    pad_value: float = 0.0
) -> str:
    """
    Process a single .npy file by creating overlapping chunks.
    
    Args:
        input_path: Path to input .npy file
        output_path: Path to save chunked data
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        pad_mode: Padding mode
        pad_value: Value for constant padding
        
    Returns:
        Success message or error message
    """
    try:
        # Load data
        data = np.load(input_path)
        
        if data.ndim != 4:
            return f"Error: Expected 4D data, got {data.ndim}D from {input_path}"
        
        # Create chunks
        chunked_data = create_overlapping_chunks(
            data, chunk_size, overlap, pad_mode, pad_value
        )
        
        # Save chunked data
        np.save(output_path, chunked_data)
        
        return f"Successfully processed {input_path} -> {output_path}"
        
    except Exception as e:
        return f"Error processing {input_path}: {str(e)}"


def process_directory(
    input_dir: str,
    output_dir: str,
    chunk_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: Tuple[int, int, int] = (32, 32, 32),
    pad_mode: str = 'constant',
    pad_value: float = 0.0,
    file_pattern: str = "*.npy"
) -> None:
    """
    Process all .npy files in a directory by creating overlapping chunks.
    
    Args:
        input_dir: Input directory containing .npy files
        output_dir: Output directory for chunked data
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        pad_mode: Padding mode
        pad_value: Value for constant padding
        file_pattern: File pattern to match
    """
    import glob
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not input_files:
        print(f"No {file_pattern} files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Process each file
    for input_file in tqdm(input_files, desc="Processing files"):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        result = process_npy_file(
            input_file, output_file, chunk_size, overlap, pad_mode, pad_value
        )
        
        if result.startswith("Error"):
            print(f"Warning: {result}")
        else:
            print(f"✓ {result}")


def visualize_chunks(
    data: Union[np.ndarray, torch.Tensor],
    chunk_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: Tuple[int, int, int] = (32, 32, 32),
    modality_idx: int = 0,
    slice_idx: int = None
) -> None:
    """
    Visualize the chunking process for a specific modality and slice.
    
    Args:
        data: Input data of shape (modalities, x, y, z)
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        modality_idx: Index of modality to visualize
        slice_idx: Index of z-slice to visualize (if None, uses middle slice)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    modalities, x, y, z = data.shape
    
    if modality_idx >= modalities:
        print(f"Modality index {modality_idx} out of range (0-{modalities-1})")
        return
    
    if slice_idx is None:
        slice_idx = z // 2
    
    if slice_idx >= z:
        print(f"Slice index {slice_idx} out of range (0-{z-1})")
        return
    
    # Extract 2D slice
    slice_data = data[modality_idx, :, :, slice_idx]
    
    # Calculate chunk boundaries
    chunk_x, chunk_y, chunk_z = chunk_size
    overlap_x, overlap_y, overlap_z = overlap
    step_x = chunk_x - overlap_x
    step_y = chunk_y - overlap_y
    
    num_chunks_x = max(1, int(np.ceil((x - overlap_x) / step_x)))
    num_chunks_y = max(1, int(np.ceil((y - overlap_y) / step_y)))
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Display the slice
    im = ax.imshow(slice_data, cmap='gray', origin='lower')
    
    # Draw chunk boundaries
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            start_x = i * step_x
            start_y = j * step_y
            
            rect = patches.Rectangle(
                (start_y, start_x), chunk_y, chunk_x,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add chunk number
            ax.text(start_y + chunk_y//2, start_x + chunk_x//2, 
                   f'{i*num_chunks_y + j}', 
                   ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Chunking Visualization\nModality {modality_idx}, Slice {slice_idx}')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Chunking utility for 4D medical imaging data")
    print("=" * 50)
    
    # Example: Create synthetic 4D data
    modalities, x, y, z = 2, 200, 200, 200
    synthetic_data = np.random.rand(modalities, x, y, z).astype(np.float32)
    
    print(f"Created synthetic data: {synthetic_data.shape}")
    
    # Create chunks
    chunked = create_overlapping_chunks(
        synthetic_data,
        chunk_size=(96, 96, 96),
        overlap=(32, 32, 32)
    )
    
    print(f"Chunked data shape: {chunked.shape}")
    
    # Reconstruct
    reconstructed = reconstruct_from_chunks(
        chunked,
        synthetic_data.shape,
        chunk_size=(96, 96, 96),
        overlap=(32, 32, 32)
    )
    
    print(f"Reconstructed data shape: {reconstructed.shape}")
    
    # Check reconstruction quality
    mse = np.mean((synthetic_data - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    if mse < 1e-10:
        print("✓ Perfect reconstruction!")
    else:
        print("⚠ Some reconstruction error detected")




import os
from PIL import Image


def delete_files_recursively(directory_path):
    """
    Recursively deletes all files in the given directory. 
    If the directory does not exist, it is created.

    Parameters
    ----------
    directory_path : str
        Path to the directory whose files should be deleted.
    """
    if os.path.exists(directory_path):
        # Walk through all subdirectories and delete files
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                except Exception as error:
                    print(f"Error deleting file {file_path}: {error}")
    else:
        # Create the directory if it does not exist
        os.makedirs(directory_path, exist_ok=True)


def save_images(image_files, output_base_dir, class_name=None):
    """
    Saves a list of image files to a specified directory, optionally under a class subdirectory.

    Parameters
    ----------
    image_files : list
        List of file-like objects or file paths representing images to be saved.
    output_base_dir : str
        Path to the base directory where images will be saved.
    class_name : str or None, optional
        Optional class subdirectory under the base directory. If provided, images are saved there.
    """
    # Build the destination directory path

    if class_name:
        target_dir = os.path.join(output_base_dir, class_name)
    else:
        target_dir = output_base_dir

    os.makedirs(target_dir, exist_ok=True)

    for image in image_files:
        imagem_pil = Image.open(image)
        path_name = os.path.join(target_dir , image.name)
        imagem_pil.save(path_name)


def is_folder_empty(path: str) -> bool:
    """
    Check if the folder contains any valid image files in its subdirectories.

    Parameters
    ----------
    path : str
        Path to the support dataset directory.

    Returns
    -------
    bool
        True if no image files (.png, .jpg, .jpeg) are found in any subdirectory, False otherwise.
    """
    if not os.path.exists(path):
        return True

    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in os.listdir(subdir_path)):
                return False

    return True
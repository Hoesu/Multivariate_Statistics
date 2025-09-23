from pathlib import Path

import numpy as np
from PIL import Image

class HumanImageDecomposer:
    """Image decomposition class using SVD, coded by human.

    Parameters
    ----------
    path: str
        Path to the image file

    Attributes
    ----------
    path: Path
        Path to the image file
    image_array: np.ndarray
        Image converted to numpy array
    image_name: str
        Image name without the file extension
    """
    def __init__(self, path: str):
        self.path = Path(path)
        self.image_array, self.image_name = self._load_image()

    def _load_image(self) -> tuple[np.ndarray, str]:
        """Load image from the path

        Returns
        -------
        image_array: np.ndarray
            Image converted to numpy array
        image_name: str
            Image name without the file extension
        """
        image = Image.open(self.path)
        image_name = self.path.stem
        image_extension = self.path.suffix

        if image_extension not in [".jpg", ".jpeg"]:
            image = image.convert("RGB")

        image_array = np.array(image, dtype=np.float64)
        return image_array, image_name

    def _save_image(self, image_array: np.ndarray, compression_target: int) -> None:
        """Save image to the result directory
        
        file name is in the format of {image_name}_human_{compression_target}.jpg

        Parameters
        ----------
        image_array: np.ndarray
            Image converted to numpy array
        compression_target: int
            Compression target
        """
        result_dir = Path.cwd() / "result"
        result_dir.mkdir(exist_ok=True)
        image = Image.fromarray(image_array)
        image.save(result_dir / f"{self.image_name}_human_{compression_target}.jpg")

    def reconstruct_image(self) -> None:
        """Reconstruct image using SVD
        
        Notes
        -----
        Images are reconstructed in the following order:
        1. separate image into R, G, B channels
        2. apply SVD to each channel
        3. truncate singular values to the compression target of 5, 20, and 50
        4. reconstruct each channel by multiplying their respective U, S (truncated), and V
        5. stack the reconstructed channels to form the final image
        6. clip the image to the range of 0-255 uint8 values
        7. save all reconstructed images to the result directory
        """
        img_r_channel = self.image_array[:, :, 0]
        img_g_channel = self.image_array[:, :, 1]
        img_b_channel = self.image_array[:, :, 2]

        u_r, spec_r, v_r = np.linalg.svd(img_r_channel)
        u_g, spec_g, v_g = np.linalg.svd(img_g_channel)
        u_b, spec_b, v_b = np.linalg.svd(img_b_channel)

        for compression_target in [5, 20, 50]:

            s_r = np.zeros((u_r.shape[0], v_r.shape[0]))
            spec_r_slice = spec_r.copy()
            spec_r_slice[compression_target:] = 0
            np.fill_diagonal(s_r, spec_r_slice)

            s_g = np.zeros((u_g.shape[0], v_g.shape[0]))
            spec_g_slice = spec_g.copy()
            spec_g_slice[compression_target:] = 0
            np.fill_diagonal(s_g, spec_g_slice)

            s_b = np.zeros((u_b.shape[0], v_b.shape[0]))
            spec_b_slice = spec_b.copy()
            spec_b_slice[compression_target:] = 0
            np.fill_diagonal(s_b, spec_b_slice)

            reconstructed_img_r_channel = u_r @ s_r @ v_r
            reconstructed_img_g_channel = u_g @ s_g @ v_g
            reconstructed_img_b_channel = u_b @ s_b @ v_b

            reconstructed_img = np.zeros_like(self.image_array)
            reconstructed_img[:, :, 0] = reconstructed_img_r_channel
            reconstructed_img[:, :, 1] = reconstructed_img_g_channel
            reconstructed_img[:, :, 2] = reconstructed_img_b_channel
            reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)

            self._save_image(
                image_array=reconstructed_img,
                compression_target=compression_target
            )
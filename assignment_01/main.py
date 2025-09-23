import yaml

from src.human import HumanImageDecomposer
from src.ai import AiImageDecomposer

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    human_image_decomposer = HumanImageDecomposer(cfg)
    human_image_decomposer.reconstruct_image()
    
    ai_image_decomposer = AiImageDecomposer(cfg)
    ai_image_decomposer.reconstruct_image()

if __name__ == "__main__":
    main()

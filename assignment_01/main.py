from src.human import HumanImageDecomposer
from src.ai import AiImageDecomposer

def main():
    human_image_decomposer = HumanImageDecomposer("assets/silksong.jpeg")
    human_image_decomposer._reconstruct_image()
    
    ai_image_decomposer = AiImageDecomposer("assets/silksong.jpeg")
    ai_image_decomposer._reconstruct_image()

if __name__ == "__main__":
    main()

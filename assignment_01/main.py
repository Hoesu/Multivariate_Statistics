from src.human import HumanImageCompressor
from src.ai import AiImageCompressor

def main():
    human_image_compressor = HumanImageCompressor("assets/silksong.jpeg")
    human_image_compressor._reconstruct_image()
    
    ai_image_compressor = AiImageCompressor("assets/silksong.jpeg")
    ai_image_compressor._reconstruct_image()

if __name__ == "__main__":
    main()

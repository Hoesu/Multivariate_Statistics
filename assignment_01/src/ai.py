import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class AiImageDecomposer:
    """AI 버전 SVD 이미지 분해 클래스"""
    
    def __init__(self, cfg: dict):
        """
        Args:
            cfg: 입력 이미지 파일 경로 (기존 워크플로우와 동일한 매개변수명)
        """
        self.image_path = cfg["image_path"]
        self.original_image = None
        self.image_array = None
        self.channels = None
        self.height = None
        self.width = None
        
        # 경로 설정
        self.assets_dir = "/home/hoesu.chung/GITHUB/Multivariate_Statistics/assignment_01/assets"
        self.output_dir = "/home/hoesu.chung/GITHUB/Multivariate_Statistics/assignment_01/result"
        
        # 과제 요구사항에 따른 특이값 개수
        self.k_values = [5, 20, 50]
        
        self._load_image()
    
    def _load_image(self):
        """이미지를 로드하고 numpy 배열로 변환"""
        try:
            # 경로 처리: 상대 경로인 경우 절대 경로로 변환
            if not os.path.isabs(self.image_path):
                # assets/ 로 시작하는 경우 전체 경로로 변환
                if self.image_path.startswith('assets/'):
                    self.image_path = os.path.join('/home/hoesu.chung/GITHUB/Multivariate_Statistics/assignment_01', self.image_path)
                else:
                    self.image_path = os.path.join(self.assets_dir, self.image_path)
            
            # PIL로 이미지 로드
            self.original_image = Image.open(self.image_path)
            
            # RGB로 변환 (RGBA나 다른 모드 처리)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            
            # numpy 배열로 변환
            self.image_array = np.array(self.original_image, dtype=np.float64)
            self.height, self.width, self.channels = self.image_array.shape
            
            print(f"이미지 로드 완료: {self.width}x{self.height}, 채널수: {self.channels}")
            
        except Exception as e:
            raise ValueError(f"이미지 로드 실패: {e}")
    
    def _apply_svd_to_channel(self, channel_matrix: np.ndarray) -> tuple:
        """단일 채널에 SVD 적용"""
        # SVD 분해: A = U * S * V^T
        U, s, Vt = np.linalg.svd(channel_matrix, full_matrices=False)
        return U, s, Vt
    
    def _reconstruct_channel(self, U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
        """상위 k개 특이값으로 채널 재구성"""
        # 상위 k개 특이값만 사용
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        
        # 채널 재구성
        reconstructed = U_k @ np.diag(s_k) @ Vt_k
        
        # 픽셀 값을 0-255 범위로 클리핑
        reconstructed = np.clip(reconstructed, 0, 255)
        
        return reconstructed
    
    def decompose_and_reconstruct(self, k_values: list) -> dict:
        """
        SVD를 사용하여 이미지를 분해하고 재구성
        
        Args:
            k_values: 사용할 특이값 개수 리스트
            
        Returns:
            dict: {k: reconstructed_image_array} 형태
        """
        results = {}
        
        for k in k_values:
            print(f"상위 {k}개 특이값으로 재구성 중...")
            
            # 각 채널별로 SVD 적용
            reconstructed_channels = []
            
            for channel_idx in range(self.channels):
                channel_matrix = self.image_array[:, :, channel_idx]
                
                # SVD 적용
                U, s, Vt = self._apply_svd_to_channel(channel_matrix)
                
                # k개 특이값으로 재구성
                reconstructed_channel = self._reconstruct_channel(U, s, Vt, k)
                reconstructed_channels.append(reconstructed_channel)
            
            # 채널들을 다시 합치기
            reconstructed_image = np.stack(reconstructed_channels, axis=2)
            reconstructed_image = reconstructed_image.astype(np.uint8)
            
            results[k] = reconstructed_image
            
            # 압축률 계산
            original_elements = self.height * self.width * self.channels
            compressed_elements = k * (self.height + self.width + 1) * self.channels
            compression_ratio = compressed_elements / original_elements * 100
            
            print(f"k={k}: 압축률 {compression_ratio:.1f}%")
        
        return results
    
    def save_results(self, results: dict, output_dir: str):
        """재구성된 이미지들을 저장"""
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 원본 파일명에서 확장자 제거
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        for k, reconstructed_image in results.items():
            # 파일명 생성: {원본이름}_ai_{k값}
            output_filename = f"{base_name}_ai_{k}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # PIL Image로 변환하여 저장
            pil_image = Image.fromarray(reconstructed_image)
            pil_image.save(output_path, 'JPEG', quality=95)
            
            print(f"저장 완료: {output_path}")
    
    def compare_images(self, results: dict):
        """원본과 재구성된 이미지들을 비교 시각화"""
        n_images = len(results) + 1  # 원본 + 재구성된 이미지들
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
        
        # 원본 이미지
        axes[0].imshow(self.original_image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 재구성된 이미지들
        for idx, (k, reconstructed) in enumerate(results.items(), 1):
            axes[idx].imshow(reconstructed)
            axes[idx].set_title(f'k={k}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_singular_values(self):
        """각 채널의 특이값 분석"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        channel_names = ['Red', 'Green', 'Blue']
        
        for channel_idx in range(self.channels):
            channel_matrix = self.image_array[:, :, channel_idx]
            _, s, _ = self._apply_svd_to_channel(channel_matrix)
            
            # 특이값 플롯
            axes[channel_idx].plot(s[:100])  # 상위 100개만 표시
            axes[channel_idx].set_title(f'{channel_names[channel_idx]} Channel Singular Values')
            axes[channel_idx].set_xlabel('Index')
            axes[channel_idx].set_ylabel('Singular Value')
            axes[channel_idx].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 각 채널의 특이값 정보 출력
        for channel_idx in range(self.channels):
            channel_matrix = self.image_array[:, :, channel_idx]
            _, s, _ = self._apply_svd_to_channel(channel_matrix)
            total_energy = np.sum(s**2)
            
            print(f"\n{channel_names[channel_idx]} 채널:")
            for k in [5, 20, 50]:
                if k <= len(s):
                    energy_ratio = np.sum(s[:k]**2) / total_energy * 100
                    print(f"  상위 {k}개 특이값으로 {energy_ratio:.1f}%의 에너지 보존")
    
    def reconstruct_image(self):
        """
        기존 워크플로우에 맞는 이미지 재구성 메소드
        main.py에서 호출되는 메인 함수
        """
        print(f"\n{'='*60}")
        print(f"AI 버전 - 처리 중인 이미지: {os.path.basename(self.image_path)}")
        print(f"{'='*60}")
        
        try:
            # 특이값 분석
            print("\n[AI - 특이값 분석]")
            self.analyze_singular_values()
            
            # SVD 분해 및 재구성
            print(f"\n[AI - SVD 분해 및 재구성]")
            results = self.decompose_and_reconstruct(self.k_values)
            
            # 결과 저장
            print(f"\n[AI - 결과 저장]")
            self.save_results(results, self.output_dir)
            
            # 이미지 비교 시각화 (선택사항)
            print(f"\n[AI - 이미지 비교 시각화]")
            self.compare_images(results)
            
            print(f"\n[AI - 완료] {os.path.basename(self.image_path)} 처리 완료!")
            
        except Exception as e:
            print(f"[AI - 오류] {e}")
            raise
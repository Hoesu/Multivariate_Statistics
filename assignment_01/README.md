# Assignment 01

Write a code in R, Python, or other computer languages that decompose a photo using SVD (Singular Value Decomposition). Apply the code to real images by using the top 5, 20, 50 eigenvalues. You should submit the written code and the results of the photo decomposition. Use LLM models such as ChatGPT and compare it to your code and results. (100 points)

## Installation

```bash
## clone the repository
git clone https://github.com/Hoesu/Multivariate_Statistics.git

## intall Astral UV on mac/linux
curl -LsSf https://astral.sh/uv/install.sh | sh

## install Astral UV on windows
powershell -c "irm https://astral.sh/uv/install.ps1 | more"

## sync the dependencies
cd ~/Multivariate_Statistics/assignment_01/
uv sync
```

## Deployment

재구성하고 싶은 이미지를 `assets` 디렉토리에 넣고, 해당 이미지의 절대 경로를 `config.yaml` 파일에 적어주세요.

```bash
uv run main.py
```

## AI Prompts

AI prompting은 **ai.py** 스크립트를 수정하는 용도로만 사용되었습니다.

Powered by Claude-4-sonnet

```bash
너는 응용통계학과 다변량통계 수업을 듣는 수강생의 코드 어시스턴트야. 이번에 수강생이 받은 과제는 다음과 같아:

Write a code in R, Python, or other computer languages that decompose a photo
using SVD (Singular Value Decomposition). Apply the code to real images by
using the top 5, 20, 50 eigenvalues. You should submit the written code and the
results of the photo decomposition. Use LLM models such as ChatGPT and
compare it to your code and results. (100 points)

해당 과제 설명문을 읽어보면 알겠지만, 너의 역할은 LLM 어시스턴트로서, 수강생의 코드 결과물과 네 코드 결과물을 비교해야 해. SVD를 활용한 코딩은 너가 알아서 하되, 입력 이미지의 경로와 출력 결과를 저장하는 디렉토리, 그리고 결과물 파일의 이름 형식은 내가 지정해줄게:

입력 이미지가 저장된 디렉토리: /home/hoesu.chung/GITHUB/Multivariate_Statistics/assignment_01/assets
출력 결과물을 저장할 디렉토리: /home/hoesu.chung/GITHUB/Multivariate_Statistics/assignment_01/result
결과물 파일 이름 형식: f'{확장자를 제외한 원본 이미지 이름}_ai_{이미지 재생성에 사용한 고유값 수}'

현재 다른 디렉토리에는 수강생이 먼저 구현한 코드가 있겠지만, 코드 복제 및 참조 방지를 위해 너 스스로의 힘으로 코드를 적어줘.
```

```bash
개선 사항 요청:

현재 코드 구조를 보면 main.py에서 어떤식으로 네 코드를 호출하여 이미지 결과를 추출하고 싶은지 힌트가 마련되어 있어. 하지만 지금 너가 만든 코드를 보면 main_ai.py를 따로 생성해서 기존의 워크플로우를 따르지 않고 있지. 내부 알고리즘은 그대로 두되, 기존에 제공된 워크플로우를 따르는 방식으로 코드를 리팩토링 해봐.
```

```bash
ai.py 스크립트 닥스트링이 마음에 안들어. human.py처럼 numpy style로 통일해줘.
```

```bash
디렉토리 관련 설정에서 절대경로로 받아서 나머지 태스크로 진행하는 것은 위험한것 같아. 만약에 다른 사용자가 이걸 사용한다면, 절대경로가 달라지기 때문에 에러가 발생할거야. configuration에서 절대경로를 넘겨받더라도, 이걸 상대경로로 유연하게 변환하여 태스크를 진행하는 방식으로 코드를 리팩토링해줘.
```
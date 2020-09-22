# 노인을 위한 손가락 궤적 기반 제스처 인식 시스템



## 0) 프로젝트 설명

### (0) video_test.py를 통해 실행

### (1) 전체 시스템 설계
![image](https://user-images.githubusercontent.com/41561652/93848409-768e4600-fce4-11ea-9b63-2d12db6ae431.png)

### (2) 손가락 검출 알고리즘
![image](https://user-images.githubusercontent.com/41561652/93848445-886fe900-fce4-11ea-8ff2-424a873d8278.png)

### (3) 제스처 매핑 알고리즘
![image](https://user-images.githubusercontent.com/41561652/93848463-945bab00-fce4-11ea-8c03-05b301cfb6fd.png)

### (4) 실행 예시
![image](https://user-images.githubusercontent.com/41561652/93849089-329c4080-fce6-11ea-87a0-708302bfd229.png)

![image](https://user-images.githubusercontent.com/41561652/93849101-3c25a880-fce6-11ea-8258-f70fc4019319.png)


## 1) 가상환경 설정

-> tensorflow                      : 1.12.0

-> opencv(opencv-contrib 없음)     : 3.4.2

=> open cv 버전 추후 더 확인 필요


## 2) 상대경로도 I/O하도록 수정
-> 입력 비디오 경로 : ./ExperimentData/*

-> 출력 텍스트, 비디오 경로 : ./ResultFile/*


## 3) yolo손 검출 가로 영역 줄임


## 4) 주의사항
yolo 기본 체크포인트는 용량상 올리지 않음(구글 드라이브에 별도 저장해둠 - darknet_weight.zip)

실험 데이터와 결과 데이터도 구글드라이브에 별도 저장(Experiment.zip / ResultFile.zip)

=> 실험 데이터 추가될 때 마다 여기에 넣고 실험 진행 

상세 내용은 별도 한글파일에 

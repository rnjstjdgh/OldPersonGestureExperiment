# 노인제스처 실험프로젝트

## 0) 프로젝트 설명

### (1) 전체 시스템 설계
![image](https://user-images.githubusercontent.com/41561652/93848040-95d8a380-fce3-11ea-868c-9055ac9d8c47.png)

### (2) 손가락 검출 알고리즘()
![image](https://user-images.githubusercontent.com/41561652/93848061-a7ba4680-fce3-11ea-891f-5d97182df59f.png)

### (3) 제스처 매핑 알고리즘
![image](https://user-images.githubusercontent.com/41561652/93848127-bef93400-fce3-11ea-9c71-14170f755630.png)

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

노인제스처 실험프로젝트


1) 가상환경 설정

-> tensorflow                      : 1.12.0

-> opencv(opencv-contrib 없음)     : 3.4.2

=> open cv 버전 추후 더 확인 필요


2) 상대경로도 I/O하도록 수정

-> 입력 비디오 경로 : ./ExperimentData/*

-> 출력 텍스트, 비디오 경로 : ./ResultFile/*


3) yolo손 검출 가로 영역 줄임



주의사항: yolo 기본 체크포인트는 용량상 올리지 않음(구글 드라이브에 별도 저장해둠 - darknet_weight.zip)

실험 데이터와 결과 데이터도 구글드라이브에 별도 저장(Experiment.zip / ResultFile.zip)

=> 실험 데이터 추가될 때 마다 여기에 넣고 실험 진행 


상세 내용은 별도 한글파일에 

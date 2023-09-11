# Yolo-Segmentation-Chess

This is a self-learning project that demonstrates how to detect chess board corner points, locate chess pieces on the chess board, and predict chess moves in real time.

## Project Flow

![Projct Flow](Doc/Flow.png)


### Setup

- Use a python virtualenv  `python3 -m venv Venv`
- `https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess.git`
- `pip install -r requirements.txt`

### Run locally

- start app using `python3 main.py` and open app on your phone with your maching ip and `5000` port-number ex. `https://192.168.4.67:5000/`
- make sure you use `https` instead of `http`. 
- Since I'm using locally produced SSL certificates, the risk warning will appear, but you can overlook it.
- grant camera permissions.
- Generate your own certificate if SSL certified doesn't work, [here](https://stackoverflow.com/a/32169444/9735841) or  [here](https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https)

♔♕♗♘♙♚♛♝♞♟♖♜

The camera captures the image of chessboard then the image is analyzed using image processing to identify the moves made by an opponent and stockfish engine calculates the best possible move.

### Method of Working

![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/02602e5b-75dd-4d3a-8651-270ed2210ced)

### Example

Input-Frame |	Predicted Mask |	Corner Points |	Warp Perspective Image |	Yolo Inference Image
-|-|-|-|-
![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/54faa9a6-410e-4f24-bff6-28783d77b23f) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/f5807df8-3f1f-4010-a11f-65c8c39f353e) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/8cac346a-e86f-466b-b0a5-2177b2f12bc9) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/2b93ddde-56c9-4f64-b9e0-cc370d584844) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/3f3a1103-3fb7-4138-a9d1-1a6a39dbece6)
![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/760fcd28-5b65-4cc9-9f5b-539a607b783d) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/a08b2889-dbee-4cf2-80d2-b2f2d523503d) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/5b94ac01-b078-4707-b050-4deafd8ca422) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/c6051bba-b14f-4be3-9914-b0993510a035) | ![image](https://github.com/Vatsalparsaniya/Yolo-Segmentation-Chess/assets/33985480/d703b6d7-0a7f-4341-beb8-e020fda5414b)








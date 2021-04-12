from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64

app = Flask(__name__)


@app.route('/')
def hello_world():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Title</title>
    </head>
    <body>
    <video id="video" width="640" height="480" autoplay></video>
    <button id="snap">Snap Photo</button>
    <canvas id="canvas" width="640" height="480"></canvas>
    </body>
    <script>

    var video = document.getElementById('video');
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            //video.src = window.URL.createObjectURL(stream);
            video.srcObject = stream;
            video.play();
        });
    }

    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    var video = document.getElementById('video');

    // Trigger photo take
    document.getElementById("snap").addEventListener("click", function() {
        context.drawImage(video, 0, 0, 200, 200);
    var request = new XMLHttpRequest();
    request.open('POST', '/submit?image=' + canvas.toDataURL(), true);
    request.send();
    });



</script>
</html>
    """


# HtmlVideoElement

@app.route('/test', methods=['GET'])
def test():
    return "hello world!"


@app.route('/submit', methods=['POST'])
def submit():
    image = request.args.get('image')

    return image


app.run(host="0.0.0.0", port=5000)

$(document).ready(function(){

    let namespace = "/videofeed";
    let video = document.getElementById("videoElement");
    let canvas = document.getElementById("canvasElement");
    let canvasContext = canvas.getContext('2d');

    // set image height and width here for processing
    canvas.width = 576
    canvas.height = 576

    photo = document.getElementById("ProcessedImage");
    borderphoto = document.getElementById("BorderImage");

    var localMediaStream = null;

    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);

    function sendSnapshot() {
        if (!localMediaStream) {return;}

        let dataURL = canvas.toDataURL('image/jpeg');
        socket.emit('Captured_Image', dataURL);
    }

    socket.on('Processed_Image',function(data){
        photo.setAttribute('src', data.image_data);
    });

    socket.on('Border_Image',function(data){
        borderphoto.setAttribute('src', data.image_data);
    });

    socket.on('connect', function() {
        console.log('Connected!');
    });

    var constraints = {
        audio: false,
            video: {
                width: { min: 400, ideal: 720, max: 1080 },
                height: { min: 400, ideal: 720, max: 1080 },
                facingMode: "environment"
            }
        };

    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUsermedia
                    || navigator.mozGetUserMedia || navigato.msGetUserMedia || navigator.oGetUserMedia;

    if(navigator.getUserMedia){
        navigator.getUserMedia(constraints, handleVideo, videoError)
    }

    function handleVideo(stream){
        video.srcObject = stream;

        if (stream) {localMediaStream = true}

        setInterval(function () {
            canvasContext.drawImage(video, 0, 0, canvas.width, canvas.height);
        }, 60);  // update image on screen every 60 ms

        setInterval(function () {
            sendSnapshot();
        }, 1500); // request for prediction on image every 1500 ms
    }

    function videoError(e){}

});


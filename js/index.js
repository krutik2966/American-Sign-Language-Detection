const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
let isVideo = false,x,y,w,h;
let model;
let trackButton = document.getElementById("videobutton");
var image,gray_img,pred,values,arr;
const ans_element = document.getElementById("ans");
const toggleButton = document.getElementById("videobutton");
const labels = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'};

console.log(videoElement);
loadModel();
function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // shows the image
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height); 
  
  if (results.multiHandLandmarks && isVideo) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                      {color: '#00FF00', lineWidth: 5});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
      console.log(landmarks);
      // const tsd = landmarks;
      // const p=tsd.dataSync();
      const arr = landmarks;
      let maxX = 0,maxY=0, minX = Infinity,minY=Infinity;
      for (let i = 0; i < 21; i++) {
          maxX = Math.max(maxX,arr[i].x);
          maxY = Math.max(maxY,arr[i].y);
          minX = Math.min(minX,arr[i].x);
          minY = Math.min(minY,arr[i].y);
      }
      maxX=Math.round(maxX*canvasElement.width);
      minX = Math.round(minX*canvasElement.width);
      maxY = Math.round(maxY*canvasElement.height);
      minY = Math.round(minY*canvasElement.height);
      // console.log(maxX,maxY,minX,minY);
      canvasCtx.beginPath();
      x = minX-20;
      y = maxY+20;
      w = maxX-minX+50;
      h = minY-maxY-50;
      canvasCtx.rect(x, y,w,h);
      // canvasCtx.fillRect(minX-20, minY-30,5,5,{color:"00FF00"});
      // canvasCtx.fillRect(minX-20, maxY+20,5,5,{color:"00FF00"});
      // canvasCtx.fillRect(maxX+30, minY-30,5,5,{color:"00FF00"});
      // canvasCtx.fillRect(maxX+30, maxY+20,5,5,{color:"00FF00"});

      canvasCtx.stroke();
      console.log(minX-20,minY-30,maxX-minX+50,maxY-minY+50);

      predictImg(results.image,minX-20,minY-30,maxX-minX+50,maxY-minY+50);
      // console.log(minX-20, maxY+20,maxX-minX+50, minY-maxY-50);
    }
  }
  canvasCtx.restore();
}


const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);


function toggleVideo() {
  if (!isVideo) {
//       // startVideo();
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await hands.send({image: videoElement});
      },
      width: canvasElement.width,
      height: canvasElement.height
    });
    camera.start();


      isVideo = true;
      toggleButton.style.color = "black";
      toggleButton.innerHTML = "Stop Detection";
      
  } else {
      isVideo=false;

      console.log("deleted");
      delete camera;
      window.location.reload();
      
  }
}

trackButton.addEventListener("click", async () => {    
    toggleVideo();
});

// model = await tf.loadLayersModel('/model/model.json');

// model.predict()

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
  }
async function loadModel() {

    model = await tf.loadLayersModel('/model/model.json');
}

function predictImg(img,x,y,width,height) {

  // image = tf.browser.fromPixels(img).mean(2)
  // .toFloat()
  image = tf.browser.fromPixels(img);
  // gray_img = image
  

  if(y+height>canvasElement.height){
    height = canvasElement.height-y;
  }

  if(x+width>canvasElement.width){
    width = canvasElement.width-x;
  }
  gray_img = image.slice([y,x,0],[height,width,-1]);

  // gray_img = tf.grayscale_to_rgb(gray_img);
  // gray_img.reshape([1,gray_img.shape[0],gray_img.shape[1]]);


  let new_gray_img = gray_img.resizeNearestNeighbor([28,28]).mean(2)
  .toFloat()
  .expandDims(0)
  .expandDims(-1);
  console.log(new_gray_img.shape);


  // var gray_arr = gray_img.dataSync();
  // let sy = y;
  // let ey = y+height;
  // let sx = x;
  // let ex = x+width;
  
  // let section = gray_arr.slice(sy, ey + 1).map(i => i.slice(sx, ex + 1));
  // new_gray_img.reshape([1,28,28,1])
  pred = model.predict(new_gray_img)
  values = pred.dataSync();
  arr = Array.from(values);
  console.log(values);
  console.log(arr);
  console.log(argMax(arr));
  ans_element.innerHTML = labels[argMax(arr)]; 
}

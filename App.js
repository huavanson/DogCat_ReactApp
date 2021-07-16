import React, { useState, useEffect } from 'react';
import { StyleSheet, Text,TextInput, View,Image,Button } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { fetch, decodeJpeg, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system'
import * as jpeg from 'jpeg-js';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

import * as ImagePicker from 'expo-image-picker';

const Stack = createStackNavigator();
const modelJson = require('./assets/model.json');
const modelWeights = require('./assets/weight.bin');


//Lib Screen

function LibScreen({navigation}) {
    const [image, setImage] = useState(null);
    const [displayText,setDisplaytext] = useState("Press button and wait")
    useEffect(() => {
      (async () => {
        if (Platform.OS !== 'web') {
          const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
          if (status !== 'granted') {
            alert('Sorry, we need camera roll permissions to make this work!');
          }
        }
      })();
    }, []);
  
    const pickImage = async () => {
      let result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });
  
      console.log(result);
  
      if (!result.canelled) {
        const source = {uri: result.uri}
        setImage(result.uri);
        // getPrediction()
      }
    };
  async function getPrediction(result){
    setDisplaytext("Loading TF")
    await tf.ready()
    console.log(result)
    setDisplaytext("Loading Model") 
    const model = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeights));
    setDisplaytext("Fetching")
    setDisplaytext("Gettingg Buffer")
    const imgB64 = await FileSystem.readAsStringAsync(result, {
      encoding: FileSystem.EncodingType.Base64,
   });
    const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;

    const raw = new Uint8Array(imgBuffer)  
    setDisplaytext("Get Image Tensor")
    const imageTensor = imageToTensor(raw)  
    const smallImg = tf.image.resizeBilinear(imageTensor, [224,224]).toFloat()
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(smallImg.div(offset))
    const x = tf.expandDims(normalized,0)
    const image_x = tf.transpose(x,[0,3,1,2])
    setDisplaytext("Getting Classification")
    const classes = ['Cat','Dog']
    const prediction = await model.predict(image_x)
    const probabilities = tf.softmax(prediction)
    console.log('softmax : ',prediction.softmax().print())
    const res = probabilities.argMax([-1]).dataSync()[0]
    const label = classes[probabilities.argMax([-1]).dataSync()[0]]
    console.log('Predicttion :',res)
    setDisplaytext(JSON.stringify(label))

  }
  function imageToTensor(rawData){
    // const TO_UINT8ARRAY = true
    
    const{width,height,data} =  jpeg.decode(rawData,true)
    const buffer = new Uint8Array(width*height*3)
    let offset = 0;
    for (let i =0; i<buffer.length; i+=3 ){
        buffer[i]= data[offset]
        buffer[i+1]= data[offset+1]
        buffer[i+2]= data[offset+2]
        offset += 4 

    }
    return tf.tensor3d(buffer, [height,width,3])
}


 
  return (
    <View style={styles.container}>
      <Text>Classification with image from library</Text>
      <View style={styles.space} />
      <Button style={styles.button} title="Pick an image from Library" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
      <View style={styles.space} />
      <Button style={styles.button} title="Predict" onPress={() => getPrediction(image)}></Button>
      <Text>{displayText}</Text>
    </View>
  );
}



 // CamScreen 
function CameraScreen({navigation}) {
  const [image, setImage] = useState(null);
  const [displayText,setDisplaytext] = useState("Press button and wait")
  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
          alert('Sorry, we need camera roll permissions to make this work!');
        }
      }
    })();
  }, []);

  const pickImage = async () => {
    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,
    });

    console.log(result);

    if (!result.canelled) {
      const source = {uri: result.uri}
      setImage(result.uri);
      // getPrediction()
    }
  };
async function getPrediction(result){
  setDisplaytext("Loading TF")
  await tf.ready()
  console.log(result)
  setDisplaytext("Loading Model") 
  const model = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeights));
  setDisplaytext("Fetching")
  setDisplaytext("Gettingg Buffer")
  const imgB64 = await FileSystem.readAsStringAsync(result, {
    encoding: FileSystem.EncodingType.Base64,
 });
  const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
  // const imageData = await response.arrayBuffer() 
  const raw = new Uint8Array(imgBuffer)  
  // const imageData = await response.arrayBuffer()
  setDisplaytext("Get Image Tensor")
  const imageTensor = imageToTensor(raw)  
  const smallImg = tf.image.resizeBilinear(imageTensor, [224,224]).toFloat()
  const offset = tf.scalar(255.0);
  const normalized = tf.scalar(1.0).sub(smallImg.div(offset))
  const x = tf.expandDims(normalized,0)
  const image_x = tf.transpose(x,[0,3,1,2])
  setDisplaytext("Getting Classification")
  const classes = ['Cat','Dog']
  const prediction = await model.predict(image_x)
  const probabilities = tf.softmax(prediction)
  console.log('softmax : ',prediction.softmax().print())
  const res = probabilities.argMax([-1]).dataSync()[0]
  const label = classes[probabilities.argMax([-1]).dataSync()[0]]
  console.log('Predicttion :',res)
  setDisplaytext(JSON.stringify(label))

}
function imageToTensor(rawData){
  // const TO_UINT8ARRAY = true
  
  const{width,height,data} =  jpeg.decode(rawData,true)
  const buffer = new Uint8Array(width*height*3)
  let offset = 0;
  for (let i =0; i<buffer.length; i+=3 ){
      buffer[i]= data[offset]
      buffer[i+1]= data[offset+1]
      buffer[i+2]= data[offset+2]
      offset += 4 

  }
  return tf.tensor3d(buffer, [height,width,3])
}



return (
  <View style={styles.container}>
    <Text>Classification with image from Cammera</Text>
    <View style={styles.space} />
    <Button style={styles.button} title="Pick an image from camera roll" onPress={pickImage} />
    {image && <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
    <View style={styles.space} />
    <Button style={styles.button} title="Predict" onPress={() => getPrediction(image)}></Button>
    <Text>{displayText}</Text>
  </View>
);
}





function App({navigation}) {
  const [url , setUrl] = useState('https://cdn.pixabay.com/photo/2020/06/30/22/34/dog-5357794__340.jpg')
  const [displayText , setDisplaytext] = useState("Press button and wait")
  const ref = React.useRef(null);
  async function getPrediction(url){
    setDisplaytext("Loading TF")
    
    await tf.ready()
    
    setDisplaytext("Loading Model") 
    const model = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeights));
    setDisplaytext("Fetching")
    const response = await fetch(url, {} , {isBinary : true})
    setDisplaytext("Gettingg Buffer")
    const imageData = await response.arrayBuffer()  
    setDisplaytext("Get Image Tensor")
    const imageTensor = imageToTensor(imageData)
    const smallImg = tf.image.resizeBilinear(imageTensor, [224,224]).toFloat()
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(smallImg.div(offset));

    const x = tf.expandDims(normalized,0)
    
    const image_x = tf.transpose(x,[0,3,1,2])

    setDisplaytext("Getting Classification")
    const classes = ['Cat','Dog']
    const prediction = await model.predict(image_x)
    const probabilities = tf.softmax(prediction)
    console.log('softmax : ',prediction.softmax().print())
    const res = probabilities.argMax([-1]).dataSync()[0]
    const label = classes[probabilities.argMax([-1]).dataSync()[0]]
    console.log('Predicttion :',res)
    setDisplaytext(JSON.stringify(label))
  }




  function imageToTensor(rawData){
      // const TO_UINT8ARRAY = true
      
      const{width,height,data} =  jpeg.decode(rawData,true)
      const buffer = new Uint8Array(width*height*3)
      let offset = 0;
      for (let i =0; i<buffer.length; i+=3 ){
          buffer[i]= data[offset]
          buffer[i+1]= data[offset+1]
          buffer[i+2]= data[offset+2]
          offset += 4 

      }
      return tf.tensor3d(buffer, [height,width,3])
  }



  return (
    <View style={styles.container}>
      <Text>You Choose an image then press the predict button. The system will classify your image as a dog or a cat. </Text>
      <Text>Try with an image from URL</Text>
      <View style={styles.space} />
      <TextInput
      style={{height : 40 ,width:"90%",borderColor:'gray', borderWidth: 1}}
      onChangeText={text => setUrl(text)}
      value = {url}
      />
      <Image style={styles.imageStyle} source={{uri:url}}></Image>
      <Button style={styles.button} title="Predict" onPress={() => getPrediction(url)}></Button>
      <Text>{displayText}</Text>
      <View style={styles.space} />
      <Button
        style={styles.button}
        title="Choose image from Library"
        onPress={() => navigation.navigate('Lib')}
      />
      <View style={styles.space} />
      <Button
        style={styles.button}
        title="Choose image from Cammera"
        onPress={() => navigation.navigate('cam')}
      />
    </View>
  );
}


function MyStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={App} />
      <Stack.Screen name="Lib" component={LibScreen} />
      <Stack.Screen name='cam' component={CameraScreen} />
    </Stack.Navigator>
  );
}

export default function A() {
  return (
    <NavigationContainer>
      <MyStack />
    </NavigationContainer>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageStyle: {
    width:200,
    height:200
   
  },
  button: {
    marginBottom: 20,
    padding: 30
  },
  space: {
    width: 30, 
    height: 30,
  },
});

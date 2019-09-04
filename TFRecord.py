import tensorflow as tf
import cv2
import numpy

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(address):
  #read an image and convert to 224,224
  img=cv2.load_image(address)
  if img is None:
    return None
  img= cv2.resize(img ,(224,224),interpolation=cv2.INTER_CUBIC)
  #convert BGR image to RGB
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  return img

def createDataRecord(out_fileName, addrs, labels):
  writer=tf.python_io.TFRecordWriter(out_fileName)
  for i in range(len(addrs)):
    img=load_image(addrs[i])
    label=labels[i]
    if img is None:
      continue
    feature = {'image_raw' : _bytes_feature(img.tostring()),
             'label' : _int64_feature(label)
            }
    example=tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
  writer.close()

def Parser(record):
  keys_to_features = {
      'image_raw' : tf.FixedLenFeature([],tf.string),
      'label' :tf.FixedLenFeature([],tf.int64)
  }
  parsed = tf.parse_single_example(record,keys_to_features)
  image= tf.decode_raw(parsed['image_raw'],tf.uint8)
  image=tf.cast(image,tf.float32)
  image=tf.reshape(image,[224,224,3])
  label=tf.cast(parsed['label'],tf.int32)

def input_fn(filenames,train,batch_size,buffer_size):
  dataset=tf.data.TFRecordDataset(filenames=filenames)
  dataset=dataset.map(parser)
  if train:
    dataset=dataset.shuffle(buffer_size=buffer_size)
    num_repeat=None
  else:
    num_repeat=1
  dataset=dataset.repeat(num_repeat)
  dataset=dataset.batch(batch_size)
  iterator=dataset.make_one_shot_iterator()
  images_batch,labels_batch=iterator.get_next()  
  x={'images':images_batch}
  y=labels_batch
  return x,y
  return image,label

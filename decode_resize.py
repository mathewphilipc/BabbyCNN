import tensorflow as tf
#print("hello world")

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
	image_string = tf.read_file(filename)
#	image_decoded = tf.image.decode_image(image_string)
# https://stackoverflow.com/questions/48340309/cant-load-files-with-tensorflow-dataset?rq=1
	image_decoded = tf.image.decode_jpeg(image_string)
	image_resized = tf.image.resize_images(image_decoded, [28, 28])
	return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/home/mathew/Desktop/NWPU-RESISC45/airplane/airplane_001.jpg",
	"/home/mathew/Desktop/NWPU-RESISC45/airplane/airplane_002.jpg",
	"/home/mathew/Desktop/NWPU-RESISC45/airplane/airplane_003.jpg",
	"/home/mathew/Desktop/NWPU-RESISC45/airplane/airplane_004.jpg",
	"/home/mathew/Desktop/NWPU-RESISC45/airplane/airplane_005.jpg",
	"/home/mathew/Desktop/NWPU-RESISC45/airport/airport_001.jpg",	
	"/home/mathew/Desktop/NWPU-RESISC45/airport/airport_002.jpg",	
	"/home/mathew/Desktop/NWPU-RESISC45/airport/airport_003.jpg",	
	"/home/mathew/Desktop/NWPU-RESISC45/airport/airport_004.jpg",	
	"/home/mathew/Desktop/NWPU-RESISC45/airport/airport_005.jpg"
	])

# filenames = tf.constant(["/home/mathew/Desktop/NWPU-RESISC45/airplane/airplane_001.jpg"])


# print(filenames[0])
# labels[i] is the label for the image in filesnames[i]
labels = tf.constant([0,0,0,0,0,1,1,1,1,1])
# labels = tf.constant([0])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
#dataset = (tf.data.Dataset.from_tensor_slices((filenames, labels))).map(_parse_function)

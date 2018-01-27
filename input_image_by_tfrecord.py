import os
import tensorflow as tf
import PIL

# define a function to list tfrecord files.

def list_tfrecord_file(file_list):
	tfrecord_list = []
	for i in range(len(file_list)):
		current_file_abs_path = os.path.abspath(file_list[i])
		if current_file_abs_path.endswirth(".tfrecord"):
			tfrecord_list.append(current_file_abs_path)
			print("found %s successfully!" % file_list[i])
		else:
			pass
	return tfrecord_list

# Traverse current directory
def tfrecord_auto_traversal():
	PATH = "/home/mathew/Desktop/NWPU-RESISC45/"
	current_folder_filename_list = os.listdir(PATH) # Change this PATH to traverse other directories if you want.
	print(current_folder_filename_list)
#	print("hellooo")
#	return 5

def main():
	print(os.listdir("/home/mathew/Desktop/NWPU-RESISC45/"))

if __name__ == "__main__":
	main()
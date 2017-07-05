# coding: utf-8
from __future__ import division
from __future__ import print_function
import argparse, os
import model

def main(args):
	trainer = model.trainer(args.coverage)

	# テキストファイルの追加
	if args.input_dir is not None:
		assert os.path.exists(args.input_dir)
		files = os.listdir(args.input_dir)
		for filename in files:
			if filename.endswith(".txt"):
				print("loading", filename)
				trainer.add_textfile(args.input_dir + "/" + filename)
	elif args.input_filename is not None:
		assert os.path.exists(args.input_filename)
		print("loading", args.input_filename)
		trainer.add_textfile(args.input_filename)
	else:
		raise Exception()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-dir", type=str, default=None, help="訓練用のテキストファイルが入っているディレクトリ.")
	parser.add_argument("-f", "--input-filename", type=str, default=None, help="訓練用のテキストファイル.")
	parser.add_argument("-m", "--model-filename", type=str, default="out", help="モデルを保存するファイルへのパス.")
	parser.add_argument("-l", "--coverage", type=int, default=8, help="後ろの何文字から素性ベクトルを作るか.")
	main(parser.parse_args())
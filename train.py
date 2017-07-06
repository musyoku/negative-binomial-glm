# coding: utf-8
from __future__ import division
from __future__ import print_function
import argparse, os
import model

def main(args):
	trainer = model.trainer(args.coverage, args.cmax, args.tmax)

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
	parser.add_argument("-cover", "--coverage", type=int, default=8, help="後ろの何文字から素性ベクトルを作るか.")
	parser.add_argument("-cmax", "--cmax", type=int, default=1, help="後ろの何文字のIDを素性にするか.")
	parser.add_argument("-tmax", "--tmax", type=int, default=4, help="後ろの何文字の文字種を素性にするか.")
	main(parser.parse_args())
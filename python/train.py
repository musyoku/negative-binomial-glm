# coding: utf-8
from __future__ import division
from __future__ import print_function
import argparse, os
import model

def main(args):
	trainer = model.trainer(args.coverage, args.cmax, args.tmax, args.sigma)

	# テキストファイルの追加
	assert os.path.exists(args.input)
	if args.input.endswith(".txt"):
		print("loading", args.input)
		trainer.add_textfile(args.input)
	else:
		files = os.listdir(args.input)
		for filename in files:
			if filename.endswith(".txt"):
				print("loading", filename)
				trainer.add_textfile(args.input + "/" + filename)

	trainer.compile()
	print("#words: %d" % trainer.get_num_words())
	print("#characters: %d" % trainer.get_num_characters())

	itr = 1
	while True:
		trainer.perform_mcmc()
		if itr % 1000 == 0:
			print("itr: {} - log likelihood: {} - MCMC acceptance: {} - precision: {}".format(itr, trainer.compute_joint_log_likelihood(), trainer.get_acceptance_rate(), trainer.compute_mean_precision(args.theta, args.max_word_length)))

		if itr % 10000 == 0:
			trainer.save(args.model_filename)
		itr += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str, default=None, help="訓練用のテキストファイルが入っているディレクトリ.")
	parser.add_argument("-m", "--model-filename", type=str, default="out/model.glm", help="モデルを保存するファイルへのパス.")
	parser.add_argument("-cover", "--coverage", type=int, default=8, help="後ろの何文字から素性ベクトルを作るか.")
	parser.add_argument("-cmax", "--cmax", type=int, default=1, help="後ろの何文字のIDを素性にするか.")
	parser.add_argument("-tmax", "--tmax", type=int, default=4, help="後ろの何文字の文字種を素性にするか.")
	parser.add_argument("-sigma", "--sigma", type=float, default=0.2, help="ランダムウォーク幅.")
	parser.add_argument("-theta", "--theta", type=float, default=0.99, help="単語長の予測の閾値.")
	parser.add_argument("-k", "--max-word-length", type=int, default=15, help="単語の最大長.")
	main(parser.parse_args())
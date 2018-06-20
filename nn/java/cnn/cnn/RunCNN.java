package com.mlearn.cnn.cnn;

import com.mlearn.cnn.dataset.Dataset;
import com.mlearn.cnn.cnn.Layer.Size;
import com.mlearn.cnn.util.ConcurenceRunner;
import com.mlearn.cnn.util.TimedTest;
import com.mlearn.cnn.cnn.CNN.LayerBuilder;

//http://www.cnblogs.com/fengfenggirl/p/cnn_implement.html
public class RunCNN {

	final static String dir = System.getProperty("user.dir") + "\\src\\main\\resources\\cnn_dataset\\";

	public static void runCnn() {
		//创建一个卷积神经网络
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));
		CNN cnn = new CNN(builder, 50);

		//导入数据集
		String fileName = dir + "train.format";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		cnn.train(dataset, 30);//
//		String modelName = dir + "model.cnn";
//		cnn.saveModel(modelName);
		dataset.clear();

		//预测
//		 CNN cnn = CNN.loadModel(modelName);
		Dataset testset = Dataset.load( dir + "test1.format", ",", 784);
		cnn.predict(testset,  dir + "test.predict");
	}

	public static void main(String[] args) {

		new TimedTest(new TimedTest.TestTask() {

			@Override
			public void process() {
				runCnn();
			}
		}, 1).test();
		ConcurenceRunner.stop();

	}

}

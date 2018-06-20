package com.mlearn.bp;

import com.tsinghuabigdata.common.utils.FileUtils;

import java.io.File;
import java.util.List;

public class MainClass2 {
    static float eta = 0.2f;
    static int init_hiddenSize = 6;
    static int max_hiddenSize = 30;
    static final int trainTime = 5000;
    final static String dir = System.getProperty("user.dir") + "\\src\\main\\resources\\data2\\";
    final static String trainfile = dir + "train.txt";
    final static String testfile = dir + "test.txt";
    final static String outputfile = dir + "out.txt";
    static List<DataNode> trainDatas;
    static List<DataNode> testDatas;

    static {
        trainDatas = DataUtil2.getDatas(new File(trainfile));
        testDatas = DataUtil2.getDatas(new File(testfile));
    }

    public static void main(String[] args) throws Exception {
        int i = 0;
        do {
            AnnClassifier annClassifier = new AnnClassifier(6, 6, 3);
            annClassifier.setTrainNodes(trainDatas);
            annClassifier.train(eta, trainTime);
            FileUtils.reBuildFile(outputfile);
            int right = 0;
            for (DataNode testData : testDatas) {
                int testType = annClassifier.test(testData);
                int realType = testData.getType();
                if (testType == realType)
                    right++;
                FileUtils.append(testData + "\t" + testType, outputfile);
            }
            FileUtils.append(right + "/" + testDatas.size(), outputfile);
            System.out.println(right + "/" + testDatas.size());
        } while (++i < 1);

//        int i = 1;
//        float accuracy = 0;
//        float testSize = testDatas.size();
//        for (float e = eta; e >= 0.001; e -= 0.001) {
//            for (int hSize = init_hiddenSize; hSize <= max_hiddenSize; hSize++) {
//                System.out.println((i++) +"\t(" + e + "," + hSize + ")");
//                int rightSize = oneTrain(e, hSize);
//                float acc = rightSize/testSize;
//                if (acc > accuracy)
//                    accuracy = acc;
//                System.out.println(rightSize + "\t" + accuracy + "\n");
//            }
//        }
//        System.out.println(i);
    }

    private static int oneTrain(float eta, int hiddenSize) {
        AnnClassifier annClassifier = new AnnClassifier(6, hiddenSize, 3);
        annClassifier.setTrainNodes(trainDatas);
        annClassifier.train(eta, trainTime);
        int right = 0;
        for (DataNode testData : testDatas) {
            int testType = annClassifier.test(testData);
            int realType = testData.getType();
            if (testType == realType)
                right++;
        }
        return right;
    }

}

package com.mlearn.bp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;

public class MainClass {
    public static void main(String[] args) throws Exception {
//		if (args.length < 5)
//		{
//			System.out.println("Usage: \n\t-train trainfile\n\t" +
//					"-test predictfile\n\t" +
//					"-sep separator, default:','\n\t" +
//					"-eta eta, default:0.5\n\t" +
//					"-iter iternum, default:5000\n\t" +
//					"-out outputfile");
//			return;
//		}
//		ConsoleHelper helper = new ConsoleHelper(args);
//		String trainfile = helper.getArg("-train", "");
//		String testfile = helper.getArg("-test", "");
//		String separator = helper.getArg("-sep", ",");
//		String outputfile = helper.getArg("-out", "");
//		float eta = helper.getArg("-eta", 0.02f);
//		int nIter = helper.getArg("-iter", 1000);
        String dir = System.getProperty("user.dir") + "\\src\\main\\resources\\data\\";
        String trainfile = dir + "train.txt";
        String testfile = dir + "test.txt";
        String separator = ",";
        String outputfile = dir + "out.txt";
        float eta = 0.02f;
        int count = 5000;

        DataUtil util = DataUtil.getInstance();
        List<DataNode> trainList = util.getDataList(trainfile, separator);
        List<DataNode> testList = util.getDataList(testfile, separator);
        BufferedWriter output = new BufferedWriter(new FileWriter(new File(
                outputfile)));
        int typeCount = util.getTypeCount();
        AnnClassifier annClassifier = new AnnClassifier(trainList.get(0).getAttribList().size(),
                trainList.get(0).getAttribList().size() + 8, typeCount);
        annClassifier.setTrainNodes(trainList);
        annClassifier.train(eta, count);
        for (int i = 0; i < testList.size(); i++) {
            DataNode test = testList.get(i);
            int type = annClassifier.test(test);
            List<Float> attribs = test.getAttribList();
            for (int n = 0; n < attribs.size(); n++) {
                output.write(attribs.get(n) + ",");
                output.flush();
            }
            String res = util.getTypeName(type);
            if (test.getType() != type)
                res += "\tfalse";
            res += "\n";
            output.write(res);
            output.flush();
        }
        output.close();
    }

}

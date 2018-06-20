package com.mlearn.bp;

import com.google.common.base.Function;
import com.tsinghuabigdata.common.utils.ListUtils;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;

public class AnnClassifier {
    private static final double mult = 0.1;
    private int mInputCount;
    private int mHiddenCount;
    private int mOutputCount;

    private List<NetworkNode> mInputNodes;
    private List<NetworkNode> mHiddenNodes;
    private List<NetworkNode> mOutputNodes;

    private float[][] mInputHiddenWeight;
    private float[][] mHiddenOutputWeight;

    private List<DataNode> trainNodes;

    public void setTrainNodes(List<DataNode> trainNodes) {
        this.trainNodes = trainNodes.subList(1,2);
        System.out.println(this.trainNodes);
    }

    public AnnClassifier(int inputCount, int hiddenCount, int outputCount) {
        trainNodes = new ArrayList<DataNode>();
        mInputCount = inputCount;
        mHiddenCount = hiddenCount;
        mOutputCount = outputCount;
        mInputNodes = new ArrayList<NetworkNode>();
        mHiddenNodes = new ArrayList<NetworkNode>();
        mOutputNodes = new ArrayList<NetworkNode>();
        mInputHiddenWeight = new float[inputCount][hiddenCount];
        mHiddenOutputWeight = new float[mHiddenCount][mOutputCount];
    }
    /**
     * 初始化
     */
    private void reset() {
        mInputNodes.clear();
        mHiddenNodes.clear();
        mOutputNodes.clear();
        for (int i = 0; i < mInputCount; i++)
            mInputNodes.add(new NetworkNode(NetworkNode.TYPE_INPUT));
        for (int i = 0; i < mHiddenCount; i++)
            mHiddenNodes.add(new NetworkNode(NetworkNode.TYPE_HIDDEN));
        for (int i = 0; i < mOutputCount; i++)
            mOutputNodes.add(new NetworkNode(NetworkNode.TYPE_OUTPUT));
        for (int i = 0; i < mInputCount; i++)
            for (int j = 0; j < mHiddenCount; j++)
                mInputHiddenWeight[i][j] = (float) (Math.random() * mult);
        for (int i = 0; i < mHiddenCount; i++)
            for (int j = 0; j < mOutputCount; j++)
                mHiddenOutputWeight[i][j] = (float) (Math.random() * mult);
    }

    public void train(float eta, int n) {
        reset();
        for (int i = 1; i <= n; i++) {
            System.out.println("n = " + i);
            for (int j = 0; j < trainNodes.size(); j++) {
                forward(trainNodes.get(j).getAttribList());
                backward(trainNodes.get(j).getType());
                updateWeights(eta);
            }
            disOut(mOutputNodes);
        }
    }

    private void disOut(List<NetworkNode> mOutputNodes) {
        System.out.println(ListUtils.map(mOutputNodes, new Function<NetworkNode, Float>() {
            @Nullable
            public Float apply(@Nullable NetworkNode networkNode) {
                return networkNode.getForwardOutputValue();
            }
        }));
    }

    /**
     * forward Propagation
     */
    private void forward(List<Float> list) {
        // input layer
        for (int k = 0; k < list.size(); k++)
            mInputNodes.get(k).setForwardInputValue(list.get(k));
        // hidden layer
        for (int j = 0; j < mHiddenCount; j++) {
            float temp = 0;
            for (int k = 0; k < mInputCount; k++)
                temp += mInputHiddenWeight[k][j]
                        * mInputNodes.get(k).getForwardOutputValue();
            mHiddenNodes.get(j).setForwardInputValue(temp);
        }
        // output layer
        for (int j = 0; j < mOutputCount; j++) {
            float temp = 0;
            for (int k = 0; k < mHiddenCount; k++)
                temp += mHiddenOutputWeight[k][j]
                        * mHiddenNodes.get(k).getForwardOutputValue();
            mOutputNodes.get(j).setForwardInputValue(temp);
        }
    }

    /**
     * back Propagation
     */
    private void backward(int type) {
        // output layer
        float error = 0f;
        for (int j = 0; j < mOutputCount; j++) {
            // 输出层计算误差把误差反向传播，这里-1（具体根据所用激活函数的下界）代表不属于，1（具体根据所用激活函数的上界）代表属于
            float result = 0;
            if (j == type)
                result = 1;
            //预测值减去真实值
            mOutputNodes.get(j).setBackwardInputValue(mOutputNodes.get(j).getForwardOutputValue() - result);
//            if (t == 399)
                error += errorCalc(result, mOutputNodes.get(j).getForwardOutputValue());
        }
        System.out.println(error);
        // hidden layer
        for (int j = 0; j < mHiddenCount; j++) {
            float temp = 0;
            for (int k = 0; k < mOutputCount; k++)
                temp += mHiddenOutputWeight[j][k] * mOutputNodes.get(k).getBackwardOutputValue();
            mHiddenNodes.get(j).setBackwardInputValue(temp);
        }
    }

    /**
     * 更新权重，每个权重的梯度都等于与其相连的前一层节点的输出乘以与其相连的后一层的反向传播的输出
     */
    private void updateWeights(float eta) {
        // 更新输入层到隐层的权重矩阵
        for (int i = 0; i < mInputCount; i++)
            for (int j = 0; j < mHiddenCount; j++)
                mInputHiddenWeight[i][j] -= eta
                        * mInputNodes.get(i).getForwardOutputValue()
                        * mHiddenNodes.get(j).getBackwardOutputValue();
        // 更新隐层到输出层的权重矩阵
        for (int i = 0; i < mHiddenCount; i++)
            for (int j = 0; j < mOutputCount; j++)
                mHiddenOutputWeight[i][j] -= eta
                        * mHiddenNodes.get(i).getForwardOutputValue()
                        * mOutputNodes.get(j).getBackwardOutputValue();
    }

    private float errorCalc(float target, float out) {
        return 0.5f * (float)Math.pow(target - out, 2);
    }

    private void displayWeight(float[][] weights) {
        for (int i = 0; i < weights.length; i++) {
            String line = "";
            for (int j = 0; j < weights[i].length; j++) {
                line += weights[i][j] + "\t";
            }
            System.out.println(line);
        }
    }

    public int test(DataNode dn) {
//        displayWeight(mInputHiddenWeight);
//        System.out.println("\n\n");
//        displayWeight(mHiddenOutputWeight);
        forward(dn.getAttribList());
        float result = 2;
        int type = 0;
        // 取最接近1的
        List<Float> a = ListUtils.map(mOutputNodes, new Function<NetworkNode, Float>() {
            @Nullable
            public Float apply(NetworkNode networkNode) {
                return networkNode.getForwardOutputValue();
            }
        });
        for (int i = 0; i < mOutputCount; i++) {
            float outValue = mOutputNodes.get(i).getForwardOutputValue();
            if (1 - outValue < result) {
                result = 1 - outValue;
                type = i;
            }
        }
        String info = "";
        if (dn.getType() != type)
            info = "false";
        System.out.println(a+"\t\t"+dn.getType()+"/"+type + "\t\t" + info);
        return type;
    }
}

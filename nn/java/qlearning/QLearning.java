package com.mlearn.qlearning;

import com.google.common.collect.Maps;

import java.util.Map;
import java.util.Random;

/**
 * Created by chenxl on 2016/9/23.
 */
public class QLearning {

    private static int curState = 1;
    private static final int terminalState = 5;

    private static final float ratio = 0.8f;
    private static final int[][] rewards = {
            {-1, -1, -1, -1,  0, -1},
            {-1, -1, -1,  0, -1, 100},
            {-1, -1, -1,  0, -1, -1},
            {-1,  0,  0, -1,  0, -1},
            { 0, -1, -1,  0, -1, 100},
            {-1,  0, -1, -1,  0, 100},
    };
    private static final Map<Integer, int[]> allowableActions;
    static {
        allowableActions = Maps.newHashMap();
        allowableActions.put(0, new int[]{4});
        allowableActions.put(1, new int[]{3,5});
        allowableActions.put(2, new int[]{3});
        allowableActions.put(3, new int[]{1,2,4});
        allowableActions.put(4, new int[]{0,3,5});
        allowableActions.put(5, new int[]{1,4,5});
    }

    private static final Random random =new Random();

    Matrix Q = new Matrix(6, 6);

    private void episode() {
        curState = random.nextInt(6);
        System.out.println("初始state：" + curState);
        do {
            int action = allowableActions.get(curState)[random.nextInt(allowableActions.get(curState).length)];
            System.out.println("select:" + action);
            int v = rewards[curState][action] + (int)(ratio * getNextMaxReward(action));
            if (v != 0) {
                Q.set(curState, action, v);
                System.out.println(Q);
            }
            curState = action;
        } while (curState != terminalState);
    }

    private int getNextMaxReward(int nextState) {
        int[] actions = allowableActions.get(nextState);
        int max = Integer.MIN_VALUE;
        for (int act : actions) {
            if (Q.get(nextState, act) > max)
                max = Q.get(nextState, act);
        }
        return max;
    }

    private String test(int initialState) {
        String path = initialState + "->";
        do {
            int[] v = Q.get(initialState);
            int next = getMaxPos(v);
            path += next + "->";
            initialState = next;
        } while (initialState != terminalState);
        return path.substring(0, path.length()-2);
    }

    private int getMaxPos(int[] v) {
        int max = Integer.MIN_VALUE, s = -1;
        for (int i = 0; i < v.length; i++) {
            if (v[i] > max) {
                max = v[i];
                s = i;
            }
        }
        return s;
    }

    public static void main(String[] args) {
        QLearning qLearning = new QLearning();
        for (int i = 0; i<400; i++) {
            qLearning.episode();
        }

        for (int i = 0; i<6; i++) {
            System.out.println(qLearning.test(i));
        }
    }

}

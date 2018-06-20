package com.mlearn.bp;

/**
 * Created by chenxl on 2016/8/22.
 */
public class TDXJ {

    public static void main(String[] args) {
        float e = 0.001f;
        float x = 0;
        float y0 = 2;
        float y1 = 0;
        float alpha = 0.1f;
        while (true) {
            x -= alpha * (2*x-3);
            y1 = x*x - 3*x + 2;
            if (Math.abs(y1-y0) < e)
                break;
            y0 = y1;
            System.out.println(y0);
        }
    }
}

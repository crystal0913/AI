package com.mlearn.qlearning;

/**
 * Created by chenxl on 2016/9/23.
 */
public class Matrix {

    int row;
    int col;
    int[][] content;

    public Matrix(int row, int col) {
        this.row = row;
        this.col = col;
        this.content = new int[row][col];
    }

    public void set(int row, int col, int value) {
        this.content[row][col] = value;
    }

    public int get(int row, int col) {
        return content[row][col];
    }

    public int[] get(int row) {
        return content[row];
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < row; i ++) {
            String rows = "";
            for (int j = 0; j < col; j++) {
                rows += String.format("%8d", content[i][j]);
            }
            sb.append(rows + "\n");
        }
        return sb.toString();
    }
}

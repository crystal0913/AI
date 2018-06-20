package com.mlearn.bp;

import com.google.common.base.Function;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.tsinghuabigdata.common.utils.FileUtils;
import com.tsinghuabigdata.common.utils.ListUtils;

import javax.annotation.Nullable;
import java.io.File;
import java.util.List;

public class DataUtil2 {
    public static List<DataNode> getDatas(File file) {
        List<String> strings = FileUtils.readFileToList(file);
        List<DataNode> datas = Lists.newArrayListWithCapacity(strings.size());
        for (String str : strings) {
            String[] sp = str.split("_");
            List<Float> floats = ListUtils.map(Splitter.on(",").split(sp[0]), new Function<String, Float>() {
                @Nullable
                public Float apply(String s) {
                    return Float.parseFloat(s)/50;
                }
            });
            datas.add(new DataNode(floats, Integer.parseInt(sp[1])));
        }
        return datas;
    }

    public static List<DataNode> getDatasQType(File file) {
        List<String> strings = FileUtils.readFileToList(file);
        List<DataNode> datas = Lists.newArrayListWithCapacity(strings.size());
        for (String str : strings) {
            if (str.startsWith("###"))
                continue;
            String s = str.substring(2, str.length() - 1);
            List<Float> floats = ListUtils.map(Splitter.on(",").split(s), new Function<String, Float>() {
                @Nullable
                public Float apply(String s) {
                    return Float.parseFloat(s);
                }
            });
            datas.add(new DataNode(floats, Integer.parseInt(str.substring(0,1))));
        }
        return datas;
    }
}

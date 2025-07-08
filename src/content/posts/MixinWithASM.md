---
title: 开启Mixin的同时使用ASM
published: 2025-07-08
description: ''
image: ''
tags: [Minecraft]
category: ''
draft: false 
lang: ''
---

最近被朋友艾特要求做一个魔改版的MWE，但是一翻源码发现MWE还在用上古时代ASM。。如果换成Mixin写将会方便很多。但是MWE的ASM宛如一辆坦克在我的世界字节码中翻来覆去。解决方法就是另开一个MixinLoader 在里面同时使用ASM。
将MixinLoader中替换以下代码
```
@NotNull
    @Override
    public String[] getASMTransformerClass() {
        return new String[]{
                MWEClassTransformer.class.getName()
        };
    }
```
之后重新将ASMLoadingPlugin重写成MixinLoader 解决完毕

> 我是不是水了一篇blog。
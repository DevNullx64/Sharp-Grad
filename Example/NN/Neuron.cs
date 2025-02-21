﻿using SharpGrad.Activation;
using SharpGrad.DifEngine;
using System.Numerics;

namespace SharpGrad
{
    public class Neuron<TType>
        where TType : IBinaryFloatingPointIeee754<TType>
    {
        public static readonly Random Rand = new();

        public readonly Variable<TType>[] Weights;
        public readonly Variable<TType> Biai;
        public readonly int Inputs;
        public readonly bool ActFunc;

        public Neuron(int inputs, bool act_func)
        {
            Weights = new Variable<TType>[inputs];
            Biai = new(TType.CreateSaturating(Rand.NextDouble()), "B");
            Inputs = inputs;
            ActFunc = act_func;
            for (int i = 0; i < inputs; i++)
            {
                Weights[i] = new(TType.CreateSaturating(Rand.NextDouble()), $"W{i}");
            }
        }
        public Value<TType> Forward(Value<TType>[] X)
        {
            Value<TType> sum = X[0] * Weights[0];
            for (int i = 1; i < Inputs; i++)
            {
                sum += X[i] * Weights[i];
            }
            sum += Biai;

            return ActFunc ? sum.ReLU() : sum;
        }

        public void Step(TType lr)
        {
            foreach (Variable<TType> w in Weights)
            {
                // Console.WriteLine(w.data);
                w.Data -= lr * w.Grad;
            }
        }
    }
}
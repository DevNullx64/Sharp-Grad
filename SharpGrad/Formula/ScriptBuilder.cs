using SharpGrad.Formula.Internal;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Formula
{
    public class ScriptBuilder<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        // Contains all the gradient dataElements.
        private List<AcceleratorBuffer<TResult>> gradients = [];
        internal IReadOnlyList<AcceleratorBuffer<TResult>> Gradients => gradients;


        private readonly List<ComputeElement<TResult>> dataElements = [];
        public IReadOnlyList<ComputeElement<TResult>> DataElements => dataElements;

        // Contains all the data dataElements.
        private readonly List<AcceleratorBuffer<TResult>> datas = [];
        internal IReadOnlyList<AcceleratorBuffer<TResult>> Datas => datas;

        private List<DataInfo<TResult>> datasInfo = [];
        public IReadOnlyList<DataInfo<TResult>> DataInfos => datasInfo;


        private readonly List<ComputeElement<TResult>> operationsElements = [];
        public IReadOnlyList<ComputeElement<TResult>> OperationsElements => operationsElements;

        // Contains all the operation dataElements.
        protected readonly List<InternalOperation<TResult>> operations = [];
        public IReadOnlyList<InternalOperation<TResult>> Operations => operations;

        // Contains all the output dataElements.
        private readonly List<AcceleratorBuffer<TResult>> outputs = [];
        internal IReadOnlyList<AcceleratorBuffer<TResult>> Outputs => outputs;


        private void DFS(ComputeElement<TResult> element)
        {
            for (int i = 0; i < element.OperandsLength; i++)
                DFS(element.GetOperand(i));
            Add(element);
        }

        public ScriptBuilder(ComputeElement<TResult> outputElement)
        {
            DFS(outputElement);
        }

        protected MultiIndex<SourceOfOperand> GetMultiIndex(ComputeElement<TResult> element)
        {
            int index;
            SourceOfOperand sourceOfOperand;

            if (element.IsData)
            {
                index = dataElements.IndexOf(element);
                sourceOfOperand = SourceOfOperand.Data;
            }
            else
            {
                index = operationsElements.IndexOf(element);
                sourceOfOperand = SourceOfOperand.Operation;
            }

            if (index < 0)
                throw new ArgumentException($"Element '{element}' not found.");
            return new MultiIndex<SourceOfOperand>(sourceOfOperand, index);
        }

        private readonly HashSet<ComputeElement<TResult>> elements = [];
        public void Add(ComputeElement<TResult> element)
        {
            if (elements.Add(element))
            {
                if (element.IsData)
                {
                    dataElements.Add(element);

                    datas.Add(element.Result.Content!);

                    if (element.IsGradiable)
                    {
                        int gradIndex = gradients.Count;
                        gradients.Add(element.Result.Gradient!);
                        datasInfo.Add(new(gradIndex));
                    }
                    else
                        datasInfo.Add(new(-1));
                }
                else
                {
                    operationsElements.Add(element);

                    MultiIndex<SourceOfOperand> leftIndex = GetMultiIndex(element.GetOperand(0));
                    MultiIndex<SourceOfOperand> rightIndex = MultiIndex<SourceOfOperand>.Empty;
                    if (element.OperandsLength > 1)
                        rightIndex = GetMultiIndex(element.GetOperand(1));

                    int outputIndex = -1;
                    int gradientIndex = -1;
                    if (element.IsOuput)
                    {
                        outputIndex = outputs.Count;
                        outputs.Add(element.Result.Content!);
                        if (element.IsGradiable)
                        {
                            gradientIndex = gradients!.Count;
                            gradients.Add(element.Result.Gradient!);
                        }
                    }

                    operations.Add(new(element.OpCode, 0, outputIndex, leftIndex, rightIndex, gradientIndex));
                }
            }
        }
    }
}
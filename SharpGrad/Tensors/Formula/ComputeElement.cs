using SharpGrad.Tensors.KPU;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors.Formula
{

    /// <summary>
    /// Represents a computation element.
    /// </summary>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    /// <remarks>
    /// This class is the base class for all computation element.
    /// </remarks>
    public abstract class ComputeElement<TResult> : IEquatable<ComputeElement<TResult>>
    where TResult : unmanaged, INumber<TResult>
    {
        #region Cache management
        private static readonly List<ComputeElement<TResult>> ExisitingElements = [];
        private static readonly Dictionary<ComputeElement<TResult>, int> ElementIndices = [];

        internal static ComputeElement<TResult> Add(ComputeElement<TResult> element)
        {
            lock (ElementIndices)
            {
                ElementIndices.Add(element, ExisitingElements.Count);
                ExisitingElements.Add(element);
                return element;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ComputeElement<TResult> Get(int index)
            => ExisitingElements[index];


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int IndexOf(ComputeElement<TResult> element)
            => ElementIndices[element];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int AddOrIndexOf(InputData<TResult> element)
        {
            if (ElementIndices.TryGetValue(element, out int result))
                return result;
            else
            {
                Add(element);
                return IndexOf(element);
            }
        }

        internal static int[] IndecesOf(ComputeElement<TResult>[] operands)
        {
            switch (operands.Length)
            {
                case 0:
                    return [];
                case 1:
                    return [AddOrIndexOf((InputData<TResult>)operands[0])];
                case 2:
                    return [AddOrIndexOf((InputData<TResult>)operands[0]), AddOrIndexOf((InputData<TResult>)operands[1])];
                default:
                    List<int> indeces = [];
                    foreach (var operand in operands)
                        indeces.Add(AddOrIndexOf((InputData<TResult>)operand));
                    return [.. indeces];
            }
        }
        #endregion


        internal readonly Result<TResult> Result;

        public Shape Shape => Result.Shape;

        public readonly OpCode OpCode;
        internal readonly int[] OperandIndices = [];
        public IReadOnlyList<ComputeElement<TResult>> Operands => OperandIndices.Select(e => Get(e)).ToList();
        public ComputeElement<TResult> GetOperand(int index) => Get(OperandIndices[index]);
        public int OperandsLength => OperandIndices.Length;

        internal ComputeElement(Result<TResult> result, OpCode opCode, params int[] operandsIndices)
        {
            Result = result;
            OpCode = opCode;
            OperandIndices = operandsIndices;
        }
        public bool IsOperation => Result.IsComputable;

        private bool isComputed;
        public bool IsComputed
        {
            get => IsOperation && isComputed;
            set
            {
                if (IsOperation && isComputed != value)
                {
                    isComputed = value;
                    if (!isComputed)
                        OnResetCompute();
                }
            }
        }

        private bool isOuput = false;
        public bool IsOuput
        {
            get => Result.IsComputable && isOuput;
            set
            {
                if (Result.IsComputable && isOuput != value)
                {
                    isOuput = value;
                    Result.IsGradiable = isOuput;
                    if (!isOuput)
                        OnResetCompute();
                }
            }
        }

        public bool IsGradiable
        {
            get => Result.IsGradiable;
            set => Result.IsGradiable = value;
        }

        public bool IsData => OperandIndices.Length == 0;

        public event Action? ResetCompute;

        public void OnResetCompute()
            => ResetCompute?.Invoke();

        protected abstract bool OperandsEquals(params int[] operandIndeces);
        public bool Equals(ComputeElement<TResult>? other)
        {
            if (other == null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            if (OpCode != other.OpCode || OperandsLength != other.OperandsLength)
                return false;

            return OperandsLength == 0
                ? Result.Equals(other.Result)
                : OperandsEquals(other.OperandIndices);
        }

        public override bool Equals(object? obj)
            => obj is ComputeElement<TResult> other && Equals(other);

        protected abstract int GetOperandsHashCode();
        public override int GetHashCode()
            => OpCode.GetHashCode() * 31 + GetOperandsHashCode();

        public static ComputeElement<TResult> operator +(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => Create<AddOp<TResult>>(left, right);
        public static ComputeElement<TResult> operator +(ComputeElement<TResult> operand)
            => operand;
        public static ComputeElement<TResult> operator -(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => Create<SubOp<TResult>>(left, right);
        public static ComputeElement<TResult> operator -(ComputeElement<TResult> operand)
            => Create<NegOp<TResult>>(operand);
        public static ComputeElement<TResult> operator *(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => Create<MulOp<TResult>>(left, right);
        public static ComputeElement<TResult> operator /(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => Create<DivOp<TResult>>(left, right);

        private static ComputeElement<TResult>? Find(OpCode opCode, params int[] operands)
            => ExisitingElements.FirstOrDefault(e => e.OpCode == opCode && e.OperandsEquals(operands));

        public static ComputeElement<TResult> Create<TOp>(ComputeElement<TResult> operand)
            where TOp : IExecUnary<TResult, TResult>
        {
            int iOperand = IndexOf(operand);
            return Find(TOp.OpCode, iOperand) ?? Add(new ComputeUnaryClass<TOp, TResult>(iOperand, false));
        }

        public static ComputeElement<TResult> Create<TOp>(ComputeElement<TResult> left, ComputeElement<TResult> right)
            where TOp : IExecBinary<TResult, TResult, TResult>
        {
            int iLeft = IndexOf(left);
            int iRight = IndexOf(right);
            return Find(TOp.OpCode, iLeft, iRight)
             ?? Add(new ComputeBinaryClass<TOp, TResult>(iLeft, iRight, false));
        }

        internal TResult[] Compute()
        {
            // 1. Get Script
            // 2. Convert to KPU script
            // 3. Execute and get result
            throw new NotImplementedException();
        }
    }
}
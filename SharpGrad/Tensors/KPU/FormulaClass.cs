using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors.KPU
{
    /// <summary>
    /// Represents the result of a computation.
    /// </summary>
    /// <typeparam name="TResult">The type of the result.</typeparam>
    internal class Result<TResult> where TResult : unmanaged
    {
        public readonly Shape Shape;
        public AcceleratorBuffer<TResult>? Content { get; set; }
        public AcceleratorBuffer<TResult>? Gradient { get; private set; }

        public bool HasContent => Content is not null;
        public bool IsGradiable
        {
            get => Gradient is not null;
            set
            {
                if (IsGradiable != value)
                    Gradient = value
                        ? KernelProcessUnit.DefaultKPU.MMU.GetBuffer<TResult>(Shape.Length)
                        : null;
            }
        }
        /// <param name="shape">The shape of the result.</param>
        /// <param name="init">True if the result should be initialized. Default is false.</param>
        public Result(Shape shape, bool isGradiable, AcceleratorBuffer<TResult>? content)
        {
            Shape = shape;
            Content = content;
            if (isGradiable)
            {
                Gradient = KernelProcessUnit.DefaultKPU.MMU.GetBuffer<TResult>(shape.Length);
            }
        }

        public Result(Shape shape, bool isGradiable = false, bool init = false)
            : this(shape, isGradiable, init ? KernelProcessUnit.DefaultKPU.MMU.GetBuffer<TResult>(shape.Length) : null)
        { }

        public void ResetGradient()
            => Gradient?.Reset();
    }

    public readonly struct MultiIndex<TEnum> where TEnum : Enum
    {
        public static readonly MultiIndex<TEnum> Empty = new(byte.MaxValue);

        private static readonly byte count = byte.CreateChecked(Enum.GetValues(typeof(TEnum)).Length);
        private static readonly byte max = (byte)(byte.MaxValue / count);

        private readonly byte @this;

        internal MultiIndex(byte rawValue)
        {
            @this = rawValue;
        }

        public MultiIndex(TEnum category, byte value)
            : this((byte)(Convert.ToByte(category) * max + value))
        { }

        public readonly TEnum Category => (TEnum)(object)(@this / max);
        public readonly byte Value => (byte)(@this % max);
    }

    public readonly struct DFSOperation<TResult>(OpCode opCode, sbyte outputIndex, MultiIndex<SourceOfOperand> leftIndex, MultiIndex<SourceOfOperand> rightIndex)
        where TResult : unmanaged, INumber<TResult>
    {
        // public readonly shape shape;

        /// <summary>
        /// The operation code.
        /// </summary>
        public readonly OpCode OpCode = opCode;

        /// <summary>
        /// Index where store the output. -1 if the output is not needed.
        /// </summary>
        public readonly sbyte OutputIndex = outputIndex;

        /// <summary>
        /// The index of the leftIndex operandIndex.
        /// </summary>
        public readonly MultiIndex<SourceOfOperand> LeftIndex = leftIndex;

        /// <summary>
        /// The index of the rightIndex operandIndex.
        /// </summary>
        public readonly MultiIndex<SourceOfOperand> RightIndex = rightIndex;
    }

    public enum SourceOfOperand : byte { Data, BCData, Operation, BCOperation }

    public class Script<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public readonly List<ComputeElement<TResult>> Datas = [];
        // public readonly List<ComputeElement<TResult>> BCData = [];
        public readonly List<ComputeElement<TResult>> Outputs = [];
        public readonly List<DFSOperation<TResult>> Operations = [];
        // public readonly List<ComputeElement<TResult>> BCOperation = [];

        internal Script()
        { }

        internal void Add(ComputeElement<TResult> element)
        {
            if (element.HasResult)
                Datas.Add(element);
            else
            {
                MultiIndex<SourceOfOperand> lIndex = GetIndex(element.OperandIndices[0]);
                MultiIndex<SourceOfOperand> rIndex = (element.OperandsLength == 2)
                    ? GetIndex(element.OperandIndices[1])
                    : MultiIndex<SourceOfOperand>.Empty;

                sbyte oIndex = -1;
                if (element.IsOuput)
                {
                    oIndex = (sbyte)Outputs.Count;
                    Outputs.Add(element);
                }

                Operations.Add(new DFSOperation<TResult>(element.OpCode, oIndex, lIndex, rIndex));
            }
        }

        private static MultiIndex<SourceOfOperand> GetIndex(int elementIndex)
            => new(ComputeElement<TResult>.Get(elementIndex).HasResult ? SourceOfOperand.Data : SourceOfOperand.Operation, (byte)elementIndex);
    }

    /// <summary>
    /// Represents a computation elementIndex.
    /// </summary>
    /// <typeparam name="TResult">The type of the computation elementIndex.</typeparam>
    /// <remarks>
    /// This class is the base class for all computation elements.
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
            if(ElementIndices.TryGetValue(element, out int result))
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
        public int OperandsLength => OperandIndices.Length;

        internal Script<TResult> DeepFirstSearch(Script<TResult> dfs)
        {
            foreach (var operand in Operands)
                operand.DeepFirstSearch(dfs);
            dfs.Add(this);
            return dfs;
        }

        private Script<TResult>? script;
        public Script<TResult> Script
            => script ??= DeepFirstSearch(new());

        internal ComputeElement(Result<TResult> result, OpCode opCode, params int[] operandsIndices)
        {
            Result = result;
            OpCode = opCode;
            OperandIndices = operandsIndices;
        }

        public abstract bool IsOuput { get; set; }
        public abstract bool IsGradiable { get; set; }
        public bool HasResult => Result.HasContent;


        public event Action? ResultChanged;

        public void OnResultChanged()
            => ResultChanged?.Invoke();

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
            => (OpCode.GetHashCode() * 31) + GetOperandsHashCode();

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
            return Find(TOp.OpCode, iOperand) ?? Add(new ComputeUnaryClass<TOp, TResult>(iOperand));
        }

        public static ComputeElement<TResult> Create<TOp>(ComputeElement<TResult> left, ComputeElement<TResult> right)
            where TOp : IExecBinary<TResult, TResult, TResult>
        {
            int iLeft = IndexOf(left);
            int iRight = IndexOf(right);
            return Find(TOp.OpCode, iLeft, iRight)
             ?? Add(new ComputeBinaryClass<TOp, TResult>(iLeft, iRight));
        }

        internal TResult[] Compute()
        {
            // 1. Get Script
            // 2. Convert to KPU script
            // 3. Execute and get result
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Represents an input data elementIndex.
    /// </summary>
    /// <typeparam name="TResult">The type of the input data elementIndex.</typeparam>
    public class InputData<TResult> : ComputeElement<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public override bool IsOuput { get => false; set { } }
        public override bool IsGradiable { get; set; }

        public TResult[] Value
        {
            get => Result.Content!.CPUData;
            set
            {
                if (!Value.SequenceEqual(value))
                {
                    Result.Content!.CPUData = value;
                    OnResultChanged();
                }
            }
        }

        internal InputData(Result<TResult> value, bool isGradiable = true)
            : base(value, OpCode.Store)
        {
            IsGradiable = isGradiable;
            Add(this);
        }

        public InputData(Shape shape, bool isGradiable, TResult[]? data)
            : this(new Result<TResult>(shape, isGradiable, data is not null))
        {
            if (data is not null)
                Value = data;
        }

        protected override bool OperandsEquals(params int[] operands)
            => Result.Equals(Get(operands[0]).Result);

        protected override int GetOperandsHashCode()
            => Result.GetHashCode();
    }

    public class OutputData<TResult> where TResult : unmanaged, INumber<TResult>
    {
        private readonly ComputeElement<TResult> BaseElement;
        public TResult[] Value => !BaseElement.HasResult
            ? BaseElement.Compute()
            : BaseElement.Result.Content!.CPUData;

        public OutputData(ComputeElement<TResult> baseElement)
        {
            BaseElement = baseElement;
            BaseElement.IsOuput = true;
        }

        public static implicit operator ComputeElement<TResult>(OutputData<TResult> output)
            => output.BaseElement;

        public static implicit operator OutputData<TResult>(ComputeElement<TResult> element)
            => new(element);
    }

    /// <summary>
    /// Represents a computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation elementIndex.</typeparam>
    internal abstract class ComputeBase<TResult> : ComputeElement<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        internal ComputeBase(Result<TResult> result, OpCode opCode, params int[] operands)
            : base(result, opCode, operands)
        {
            foreach (var operand in operands)
                Get(operand).ResultChanged += OnResultChangedHandler;
            Add(this);
        }

        public override bool IsOuput { get; set; }
        public override bool IsGradiable { get => false; set { } }

        private void OnResultChangedHandler()
        {
            if (IsOuput)
            {
                if (HasResult)
                    Result.Content = null;
                else
                    return;
            }
            OnResultChanged();
        }
    }

    /// <summary>
    /// Represents a unary computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation elementIndex.</typeparam>
    internal class ComputeUnaryClass<TOp, TResult>(int operandIndex) :
        ComputeBase<TResult>(new(TOp.ResultingShape(Get(operandIndex).Shape)), TOp.OpCode, operandIndex)
        where TOp : IExecUnary<TResult, TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeElement<TResult> Operand
            => Operands[0];

        protected override bool OperandsEquals(params int[] operandIndeces)
            => OperandIndices[0] == operandIndeces[0];

        protected override int GetOperandsHashCode()
            => Operand.GetHashCode();
    }

    /// <summary>
    /// Represents a binary computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation elementIndex.</typeparam>
    internal class ComputeBinaryClass<TOp, TResult>(int leftIndex, int rightIndex) :
        ComputeBase<TResult>(new(TOp.ResultingShape(Get(leftIndex).Shape, Get(rightIndex).Shape)), TOp.OpCode, leftIndex, rightIndex)
        where TOp : IExecBinary<TResult, TResult, TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeElement<TResult> Left => Get(OperandIndices[0]);
        public ComputeElement<TResult> Right => Get(OperandIndices[1]);

        protected override bool OperandsEquals(params int[] operandIndeces)
        {
            bool result = OperandIndices[0] == operandIndeces[0] && OperandIndices[1] == operandIndeces[1];
            if(!result && (OpCode & OpCode.IsCommutative) == OpCode.IsCommutative)
                result = OperandIndices[0] == operandIndeces[1] && OperandIndices[1] == operandIndeces[0];
            return result;
        }

        protected override int GetOperandsHashCode()
        {
            int hash = Operands[0].GetHashCode();
            if ((OpCode & OpCode.IsCommutative) == OpCode.IsCommutative)
                for (int i = 1; i < OperandsLength; i++)
                    hash ^= Operands[i].GetHashCode();
            else
                for (int i = 1; i < OperandsLength; i++)
                    hash = hash * 31 + Operands[i].GetHashCode();
            return HashCode.Combine(OpCode, hash);
        }
    }

    public static class TestThis
    {
        public static void Main()
        {

        }
    }
}
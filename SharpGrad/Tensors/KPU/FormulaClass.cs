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
    /// <param name="Shape">The shape of the result.</param>
    /// <param name="init">True if the result should be initialized. Default is false.</param>
    internal class Result<TResult>(Shape Shape, bool isGradiable, AcceleratorBuffer<TResult>? content)
        where TResult : unmanaged
    {
        public readonly Shape Shape = Shape;
        public AcceleratorBuffer<TResult>? Content { get; set; } = content;
        public AcceleratorBuffer<TResult>? Gradient { get; set; } = isGradiable
            ? KernelProcessUnit.DefaultKPU.MMU.GetBuffer<TResult>(Shape.Length)
            : null;

        public bool HasContent => Content is not null;
        public readonly bool IsGradiable = isGradiable;

        public Result(Shape shape, bool isGradiable = false, bool init = false)
            : this(shape, isGradiable, init ? KernelProcessUnit.DefaultKPU.MMU.GetBuffer<TResult>(shape.Length) : null)
        { }
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

    internal readonly struct DFSOperation<TResult>(OpCode opCode, sbyte outputIndex, MultiIndex<DFS<TResult>.SourceOfOperand> leftIndex, MultiIndex<DFS<TResult>.SourceOfOperand> rightIndex)
        where TResult : unmanaged, INumber<TResult>
    {
        // public readonly Shape Shape;

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
        public readonly MultiIndex<DFS<TResult>.SourceOfOperand> LeftIndex = leftIndex;

        /// <summary>
        /// The index of the rightIndex operandIndex.
        /// </summary>
        public readonly MultiIndex<DFS<TResult>.SourceOfOperand> RightIndex = rightIndex;
    }

    public readonly struct DFS<TResult> : IEnumerable<DFSOperation<TResult>>
        where TResult : unmanaged, INumber<TResult>
    {
        public enum SourceOfOperand : byte { Data, BCData, Operation, BCOperation }
        internal readonly AcceleratorBuffer<TResult>[] Datas;
        internal readonly sbyte[] GradiableIndex;
        internal readonly AcceleratorBuffer<TResult>[] Gradients;
        //public readonly DFSData<TResult>[] BCDatas = bcDatas;
        internal readonly DFSOperation<TResult>[] Operations;

        internal DFS(AcceleratorBuffer<TResult>[] datas, sbyte[] gradiableIndex, AcceleratorBuffer<TResult>[] gradients, DFSOperation<TResult>[] operations)
        {
            if (datas.Length != gradiableIndex.Length)
                throw new ArgumentException($"The length of {nameof(datas)}({datas.Length}) and {nameof(gradiableIndex)}({gradiableIndex.Length}) must be the same.");
            int maxGradIndex = -1;
            foreach (var gIndex in gradiableIndex)
            {
                if (gIndex != -1)
                {
                    if (gIndex > maxGradIndex)
                        maxGradIndex = gIndex;
                    if (datas[gIndex].Length != gradients[gIndex].Length)
                        throw new ArgumentException($"The length of {nameof(datas)}({datas[gIndex].Length}) and {nameof(gradients)}({gradients[gIndex].Length}) must be the same.");
                }
            }
            if (maxGradIndex != gradients.Length - 1)
                throw new ArgumentException($"The maximun index in {nameof(gradiableIndex)}({maxGradIndex}) must be consistent with the length of {nameof(gradients)}({gradients.Length}).");

            Datas = datas;
            GradiableIndex = gradiableIndex;
            Gradients = gradients;
            Operations = operations;
        }

        //public readonly DFSOperation<TResult>[] BCOperations = bcOperations;

        IEnumerator<DFSOperation<TResult>> IEnumerable<DFSOperation<TResult>>.GetEnumerator()
            => Operations.AsEnumerable().GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
            => Operations.AsEnumerable().GetEnumerator();

        public static DFS<TResult> CreateFrom(ComputeElement<TResult> element)
        {
            List<AcceleratorBuffer<TResult>> datas = [];
            List<sbyte> gradients = [];
            List<AcceleratorBuffer<TResult>> grads = [];
            List<DFSOperation<TResult>> operations = [];

            return new DFS<TResult>([.. datas], [.. gradients], [.. grads], [.. operations]);
        }
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
                    return indeces.ToArray();
            }
        }
        #endregion


        internal class DFS
        {
            public List<int> DataIndices { get; } = [];
            // public List<ComputeElement<TResult>> BCData { get; } = [];
            public List<int> OperationIndeces { get; } = [];
            // public List<ComputeElement<TResult>> BCOperation { get; } = [];

            internal void Add(int elementIndex)
            {
                if (Get(elementIndex).HasResult)
                {
                    if (!DataIndices.Contains(elementIndex))
                        DataIndices.Add(elementIndex);
                }
                else if (!OperationIndeces.Contains(elementIndex))
                {
                    if (!DataIndices.Contains(elementIndex))
                    OperationIndeces.Add(elementIndex);
                }
            }

            internal void Add(ComputeElement<TResult> element)
                => Add(IndexOf(element));

            private static MultiIndex<DFS<TResult>.SourceOfOperand> GetIndex(int elementIndex)
                => new(Get(elementIndex).HasResult ? DFS<TResult>.SourceOfOperand.Data : DFS<TResult>.SourceOfOperand.Operation, (byte)elementIndex);

            internal DFS<TResult> ToStruct()
            {
                AcceleratorBuffer<TResult>[] datas = new AcceleratorBuffer<TResult>[DataIndices.Count];
                List<AcceleratorBuffer<TResult>> gradients = [];
                sbyte[] gradiableIndex = new sbyte[DataIndices.Count];
                sbyte gIndex = 0;
                for (int i = 0; i < DataIndices.Count; i++)
                {
                    var data = Get(DataIndices[i]);
                    datas[i] = data.Result.Content!;
                    if (data.IsGradiable)
                    {
                        gradiableIndex[i] = ++gIndex;
                        gradients[i] = data.Result.Gradient!;
                    }
                    else
                    {
                        gradiableIndex[i] = (sbyte)-1;
                    }
                }
                //var grads = DataIndices.Where(e => e.IsGradiable).Select(e => new Result<TResult>(e.Shape));
                List<Result<TResult>> outputs = [];
                DFSOperation<TResult>[] operations = new DFSOperation<TResult>[OperationIndeces.Count];
                int iLast = OperationIndeces.Count - 1;
                for (int i = 0; i <= iLast; i++)
                {
                    var e = Get(OperationIndeces[i]);

                    sbyte oIndex;
                    if (e.IsOuput || i == iLast)
                    {
                        oIndex = (sbyte)outputs.Count;
                        outputs.Add(e.Result);
                    }
                    else
                        oIndex = -1;

                    MultiIndex<DFS<TResult>.SourceOfOperand> lIndex = GetIndex(e.OperandIndices[0]);
                    MultiIndex<DFS<TResult>.SourceOfOperand> rIndex = (e.OperandsLength == 2)
                         ? GetIndex(e.OperandIndices[1])
                         : MultiIndex<DFS<TResult>.SourceOfOperand>.Empty;
                    operations[i] = new DFSOperation<TResult>(e.OpCode, oIndex, lIndex, rIndex);
                }

                return new DFS<TResult>(datas, gradiableIndex, null, operations);
            }
        }

        internal readonly Result<TResult> Result;
        public Shape Shape => Result.Shape;

        public readonly OpCode OpCode;
        protected readonly int[] OperandIndices = [];
        public IReadOnlyList<ComputeElement<TResult>> Operands => OperandIndices.Select(e => Get(e)).ToList();
        public int OperandsLength => OperandIndices.Length;

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

        internal abstract DFS DeepFirstSearch(DFS dfs);
        public DFS<TResult> DeepFirstSearch() => DeepFirstSearch(new()).ToStruct();

        protected abstract bool OperandsEquals(params int[] operands);
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
            // 1. Get DFS
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
        { }

        internal override DFS DeepFirstSearch(DFS dfs)
        {
            dfs.Add(this);
            return dfs;
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

        internal override DFS DeepFirstSearch(DFS dfs)
        {
            foreach (var operand in Operands)
                operand.DeepFirstSearch(dfs);
            dfs.Add(this);
            return dfs;
        }

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
        public ComputeElement<TResult> Operand => Operands[0];

        protected override bool OperandsEquals(params int[] other)
            => OperandIndices[0] == other[0];

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
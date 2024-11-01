using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

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
        /// The index of the left operand.
        /// </summary>
        public readonly MultiIndex<DFS<TResult>.SourceOfOperand> LeftIndex = leftIndex;

        /// <summary>
        /// The index of the right operand.
        /// </summary>
        public readonly MultiIndex<DFS<TResult>.SourceOfOperand> RightIndex = rightIndex;
    }

    internal readonly struct DFS<TResult> : IEnumerable<DFSOperation<TResult>>
        where TResult : unmanaged, INumber<TResult>
    {
        public enum SourceOfOperand : byte { Data, BCData, Operation, BCOperation }
        public readonly AcceleratorBuffer<TResult>[] Datas;
        public readonly sbyte[] GradiableIndex;
        public readonly AcceleratorBuffer<TResult>[] Gradients;
        //public readonly DFSData<TResult>[] BCDatas = bcDatas;
        public readonly DFSOperation<TResult>[] Operations;

        public DFS(AcceleratorBuffer<TResult>[] datas, sbyte[] gradiableIndex, AcceleratorBuffer<TResult>[] gradients, DFSOperation<TResult>[] operations)
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
                    if(datas[gIndex].Length != gradients[gIndex].Length)
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

        public IEnumerator<DFSOperation<TResult>> GetEnumerator()
            => Operations.AsEnumerable().GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();

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
    /// Represents a computation element.
    /// </summary>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    /// <remarks>
    /// This class is the base class for all computation elements.
    /// </remarks>
    public abstract class ComputeElement<TResult>(OpCode opCode, params ComputeElement<TResult>[] operands)
        where TResult : unmanaged, INumber<TResult>
    {
        public class DFS
        {
            public List<ComputeElement<TResult>> Data { get; } = [];
            // public List<ComputeElement<TResult>> BCData { get; } = [];
            public List<ComputeElement<TResult>> Operation { get; } = [];
            // public List<ComputeElement<TResult>> BCOperation { get; } = [];

            internal void Add(ComputeElement<TResult> element)
            {
                if (!Data.Contains(element))
                    Data.Add(element);
            }

            internal void Add(ComputeBase<TResult> element)
            {
                if (element.HasResult)
                    Add((ComputeElement<TResult>)element);
                else if (!Operation.Contains(element))
                    Operation.Add(element);
            }

            private MultiIndex<DFS<TResult>.SourceOfOperand> GetIndex(ComputeElement<TResult> element)
            {
                if(element.HasResult)
                {
                    int index = Data.IndexOf(element);
                    if (index == -1)
                        throw new ArgumentException($"The element {element} should be in the data list.");
                    return new MultiIndex<DFS<TResult>.SourceOfOperand>(DFS<TResult>.SourceOfOperand.Data, (byte)index);
                }
                else
                {
                    int index = Operation.IndexOf(element);
                    if (index == -1)
                        throw new ArgumentException($"The element {element} should be in the operation list.");
                    return new MultiIndex<DFS<TResult>.SourceOfOperand>(DFS<TResult>.SourceOfOperand.Operation, (byte)index);
                }
            }

            internal DFS<TResult> ToStruct()
            {
                AcceleratorBuffer<TResult>[] datas = new AcceleratorBuffer<TResult>[Data.Count];
                List<AcceleratorBuffer<TResult>> gradients = [];
                sbyte[] gradiableIndex = new sbyte[Data.Count];
                sbyte gIndex = 0;
                for (int i = 0; i < Data.Count; i++)
                {
                    datas[i] = Data[i].GetResult().Content!;
                    if (Data[i].IsNeedGrad)
                    {
                        gradiableIndex[i] = ++gIndex;
                        gradients[i] = Data[i].GetResult().Gradient!;
                    }
                    else
                    {
                        gradiableIndex[i] = (sbyte)-1;
                    }
                }
                //var grads = Data.Where(e => e.IsGradiable).Select(e => new Result<TResult>(e.Shape));
                List<Result<TResult>> outputs = [];
                DFSOperation<TResult>[] operations = new DFSOperation<TResult>[Operation.Count];
                int iLast = Operation.Count - 1;
                for (int i = 0; i <= iLast; i++)
                {
                    var e = Operation[i];

                    sbyte oIndex;
                    if (e.IsOuput || i == iLast)
                    {
                        oIndex = (sbyte)outputs.Count;
                        outputs.Add(e.GetResult());
                    }
                    else
                        oIndex = -1;

                    MultiIndex<DFS<TResult>.SourceOfOperand> lIndex = GetIndex(e.Operands[0]);
                    MultiIndex<DFS<TResult>.SourceOfOperand> rIndex = (e.Operands.Length == 2)
                         ? GetIndex(e.Operands[1])
                         : MultiIndex<DFS<TResult>.SourceOfOperand>.Empty;
                    operations[i] = new DFSOperation<TResult>(e.OpCode, oIndex, lIndex, rIndex);
                }

                return new DFS<TResult>(datas, gradiableIndex, null, operations);
            }
        }

        internal abstract Result<TResult> GetResult();

        public Shape Shape => GetResult().Shape;
        public readonly OpCode OpCode = opCode;
        public readonly ComputeElement<TResult>[] Operands = operands;

        public abstract bool IsOuput { get; set; }
        public abstract bool IsNeedGrad { get; set; }
        public bool HasResult => GetResult().HasContent;

        public event Action? ResultChanged;

        public void OnResultChanged()
            => ResultChanged?.Invoke();

        internal abstract DFS DeepFirstSearch(DFS dfs);
        public DFS DeepFirstSearch() => DeepFirstSearch(new());

        public static ComputeElement<TResult> operator +(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => new ComputeBinaryClass<AddOp<TResult>, TResult>(left, right);
        public static ComputeElement<TResult> operator +(ComputeElement<TResult> operand)
            => operand;
        public static ComputeElement<TResult> operator -(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => new ComputeBinaryClass<SubOp<TResult>, TResult>(left, right);
        public static ComputeElement<TResult> operator -(ComputeElement<TResult> operand)
            => new ComputeUnaryClass<NegOp<TResult>, TResult>(operand);
        public static ComputeElement<TResult> operator *(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => new ComputeBinaryClass<MulOp<TResult>, TResult>(left, right);
        public static ComputeElement<TResult> operator /(ComputeElement<TResult> left, ComputeElement<TResult> right)
            => new ComputeBinaryClass<DivOp<TResult>, TResult>(left, right);
    }

    /// <summary>
    /// Represents an input data element.
    /// </summary>
    /// <typeparam name="TResult">The type of the input data element.</typeparam>
    internal class InputData<TResult>(Result<TResult> value, bool isGradiable = true) : ComputeElement<TResult>(OpCode.Store)
        where TResult : unmanaged, INumber<TResult>
    {
        public override bool IsOuput { get => false; set { } }
        public override bool IsNeedGrad { get; set; } = isGradiable;

        private Result<TResult> result = value;

        internal override Result<TResult> GetResult() => result;

        public TResult[] Value
        {
            get => result.Content.CPUData;
            set
            {
                if(!Value.SequenceEqual(value))
                {
                    result.Content!.CPUData = value;
                    OnResultChanged();
                }
            }
        }

        public InputData(Shape shape, bool isGradiable, TResult[]? data)
            : this(new Result<TResult>(shape, isGradiable, data is not null))
        { }

        internal override DFS DeepFirstSearch(DFS dfs)
        {
            dfs.Add(this);
            return dfs;
        }

        public override int GetHashCode()
            => Value.GetHashCode();
    }

    /// <summary>
    /// Represents a computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    internal abstract class ComputeBase<TResult> : ComputeElement<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeBase(OpCode opCode, params ComputeElement<TResult>[] operands)
            : base(opCode, operands)
        {
            foreach (var operand in operands)
                operand.ResultChanged += OnResultChangedHandler;
        }

        public override bool IsOuput { get; set; }
        public override bool IsNeedGrad { get => false; set { } }

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
                    GetResult().Content = null;
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
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    internal class ComputeUnaryClass<TOp, TResult>(ComputeElement<TResult> operand) : ComputeBase<TResult>(TOp.OpCode, operand)
        where TOp : IExecUnary<TResult, TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeElement<TResult> Operand => Operands[0];

        private readonly Result<TResult> result = new(TOp.ResultingShape(operand.Shape));

        internal override Result<TResult> GetResult()
        {
            return result;
        }
    }

    /// <summary>
    /// Represents a binary computation operation.
    /// </summary>
    /// <typeparam name="TOp">The type of the operation.</typeparam>
    /// <typeparam name="TResult">The type of the computation element.</typeparam>
    internal class ComputeBinaryClass<TOp, TResult>(ComputeElement<TResult> left, ComputeElement<TResult> right) : ComputeBase<TResult>(TOp.OpCode, left, right)
        where TOp : IExecBinary<TResult, TResult, TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        public ComputeElement<TResult> Left => Operands[0];
        public ComputeElement<TResult> Right => Operands[1];

        private readonly Result<TResult> result = new(TOp.ResultingShape(left.Shape, right.Shape));

        internal override Result<TResult> GetResult()
        {
            return result;
        }
    }

    public static class TestThis
    {
        public static void Main()
        {

        }
    }
}

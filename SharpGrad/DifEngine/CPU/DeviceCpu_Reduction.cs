//#define MP
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public partial class DeviceCpu
    {
        // Cache for the ExecuteReductionForward delegates
        private readonly Dictionary<
            (KindReduction kind, Type Type),
            Action<KindReduction, Value, Value, Dimension>> cacheExecuteReductionForwards = [];
        private readonly Dictionary<
            (KindReduction kind, Type Type),
            Action<KindReduction, Value, Value, Dimension[]>> cacheExecuteReductionForwardsMulti = [];

        /// <summary>
        /// Executes the forward pass of a reduction operation on the given input and stores the result in the given output.
        /// </summary>
        /// <param name="kind">The kind of reduction operation to execute.</param>
        /// <param name="input">The input Value.</param>
        /// <param name="output">The output Value.</param>
        /// <param name="reduceDim">The dimension to reduce over.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the forward pass for different element types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteReductionForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteReductionForward(KindReduction kind, Value input, Value output, Dimension reduceDim)
        {
            // Check if a cached delegate exists for the reduction kind and element type
            Type elementType = output.ElementType;
            if (!cacheExecuteReductionForwards.TryGetValue((kind, elementType), out Action<KindReduction, Value, Value, Dimension>? action))
            {
                // Get the MethodInfo for the instance method ExecuteReductionForward<elementType>(KindReduction, Value, Value, Dimension)
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteReductionForward),
                    1,
                    typeof(KindReduction), typeof(Value), typeof(Value), typeof(Dimension))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteReductionForward)} not found.");
                method = method.MakeGenericMethod(elementType);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<KindReduction, Value, Value, Dimension>>(this);

                // Cache the delegate for future use
                cacheExecuteReductionForwards[(kind, elementType)] = action;
            }
            // Call the generic ExecuteReductionForward<T> method
            action(kind, input, output, reduceDim);
        }

        /// <summary>
        /// Executes the forward pass of a reduction operation on the given input and stores the result in the given output.
        /// </summary>
        /// <param name="kind">The kind of reduction operation to execute.</param>
        /// <param name="input">The input Value.</param>
        /// <param name="output">The output Value.</param>
        /// <param name="reduceDims">The dimensions to reduce over.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the forward pass for different element types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteReductionForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteReductionForward(KindReduction kind, Value input, Value output, Dimension[] reduceDims)
        {
            if (reduceDims.Length == 1)
            {
                ExecuteReductionForward(kind, input, output, reduceDims[0]);
                return;
            }

            Type elementType = output.ElementType;
            if (!cacheExecuteReductionForwardsMulti.TryGetValue((kind, elementType), out Action<KindReduction, Value, Value, Dimension[]>? action))
            {
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteReductionForward),
                    1,
                    typeof(KindReduction), typeof(Value), typeof(Value), typeof(Dimension[]))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteReductionForward)} not found.");
                method = method.MakeGenericMethod(elementType);
                action = method.CreateDelegate<Action<KindReduction, Value, Value, Dimension[]>>(this);
                cacheExecuteReductionForwardsMulti[(kind, elementType)] = action;
            }
            action(kind, input, output, reduceDims);
        }

        private void ExecuteReductionForward<TType>(KindReduction kind, Value untypedInput, Value untypedOutput, Dimension[] reduceDims)
            where TType : struct, INumber<TType>
        {
            TType[] currentInput = untypedInput.GetInitializedDataSpan<TType>().ToArray();
            Shape currentShape = untypedInput.Shape;

            KindBinary baseOp = kind.GetBaseOperation();
            Func<TType, TType, TType> operation = BinaryOperations.GetKindForwardDelegate<TType>(baseOp);
            TType neutral = ((KindGraphNode)baseOp).GetNeutralElement<TType>();

            for (int d = reduceDims.Length - 1; d >= 0; d--)
            {
                Dimension reduceDim = reduceDims[d];
                Shape reducedShape = currentShape.Remove(reduceDim);
                bool isFinal = d == 0;
                TType[] currentOutput = isFinal
                    ? untypedOutput.GetOrInitializeData<TType>().ToArray()
                    : new TType[reducedShape.Size];

                FillSpan(currentOutput.AsSpan(), neutral);
                Reduce(currentInput, currentShape, currentOutput, reduceDim, operation);

                if (isFinal)
                {
                    currentOutput.CopyTo(untypedOutput.GetInitializedDataSpan<TType>());
                }

                currentInput = currentOutput;
                currentShape = reducedShape;
            }
        }

        private void ExecuteReductionForward<TType>(KindReduction kind, Value untypedInput, Value untypedOutput, Dimension reduceDim)
            where TType : struct, INumber<TType>
        {
            untypedOutput.InitializeData();

            TType[] input = untypedInput.GetInitializedDataSpan<TType>().ToArray();
            TType[] output = untypedOutput.GetInitializedDataSpan<TType>().ToArray();

            KindBinary baseOp = kind.GetBaseOperation();
            Func<TType, TType, TType> operation = BinaryOperations.GetKindForwardDelegate<TType>(baseOp);

            FillSpan(output.AsSpan(), ((KindGraphNode)baseOp).GetNeutralElement<TType>());
            Reduce(input, untypedInput.Shape, output, reduceDim, operation);
            output.CopyTo(untypedOutput.GetInitializedDataSpan<TType>());
        }

        public static void FillArray<TType>(TType[] output, TType neutralElement)
            where TType : struct, INumber<TType>
        {
            FillSpan(output.AsSpan(), neutralElement);
        }

        internal void Reduce<TType>(TType[] input, Shape inputShape, TType[] output, Dimension reduceDim, Func<TType, TType, TType> operation)
         where TType : struct, INumber<TType>
        {
            Shape outputShape = inputShape.Remove(reduceDim);
            int dimIndex = inputShape.IndexOf(reduceDim);
            if (dimIndex < 0)
            {
                throw new ArgumentException($"Dimension {reduceDim} not found in input shape {inputShape}.");
            }

            int dimSize = reduceDim.Size;
            int dimStride = inputShape.GetStride(dimIndex);
            int outputLength = output.Length;

            ParallelFor(0, outputLength, range =>
            {
                for (int iOutput = range.Item1; iOutput < range.Item2; iOutput++)
                {
                    int iInputBase = inputShape.GetLinearIndex(iOutput, outputShape, false);

                    TType accumulator = output[iOutput];
                    int iInputEnd = iInputBase + dimSize * dimStride;
                    for (int iInput = iInputBase; iInput < iInputEnd; iInput += dimStride)
                    {
                        accumulator = operation(accumulator, input[iInput]);
                    }
                    output[iOutput] = accumulator;
                }
            });
        }

        // Cache for the ExecuteReductionBackward delegates
        private readonly Dictionary<
            (KindReduction kind, Type Value, Type Gradient),
            Action<KindReduction, Value, Value, Dimension>> cacheExecuteReductionBackwards = [];

        /// <summary>
        /// Executes the backward pass of a reduction operation on the given input and output Values.
        /// </summary>
        /// <param name="kind">The kind of reduction operation.</param>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <param name="reduceDim">The dimension that was reduced.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the ExecuteReductionBackward method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteReductionBackward(KindReduction kind, Value input, Value output, Dimension reduceDim)
        {
            // Call the generic ExecuteReductionBackward<T, G>(KindReduction, Value, Value, Dimension) method
            (KindReduction, Type Value, Type Gradient) key = (kind, output.ElementType, output.Grad.ElementType);
            if (!cacheExecuteReductionBackwards.TryGetValue(key, out Action<KindReduction, Value, Value, Dimension>? action))
            {
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteReductionBackward),
                    2,
                    typeof(KindReduction), typeof(Value), typeof(Value), typeof(Dimension))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteReductionBackward)} not found.");
                method = method.MakeGenericMethod(key.Value, key.Gradient);
                action = method.CreateDelegate<Action<KindReduction, Value, Value, Dimension>>(this);
                cacheExecuteReductionBackwards[key] = action;
            }
            action(kind, input, output, reduceDim);
        }

        /// <summary>
        /// Executes the backward pass of a reduction operation on the given input and output Values.
        /// </summary>
        /// <typeparam name="TType">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TGrad">The type of the gradients. Must be a struct that implements <see cref="IFloatingPointIeee754{G}"/>.</typeparam>
        /// <param name="kind">The kind of reduction operation.</param>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <param name="reduceDim">The dimension that was reduced.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the backward method is not found.</exception>
        private void ExecuteReductionBackward<TType, TGrad>(KindReduction kind, Value untypedInput, Value untypedOutput, Dimension reduceDim)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            untypedInput.InitializeGrad<TGrad>();

            KindBinary baseOp = kind.GetBaseOperation();
            KindBinary inverseOp = (KindBinary)((KindGraphNode)baseOp).GetInverse();

            Func<TType, TType, TType> inverseOperation = BinaryOperations.GetKindForwardDelegate<TType>(inverseOp);
            Func<TType, TType, TGrad, TGrad> backwardRight = BinaryOperations.GetKindBackwardRightDelegate<TType, TGrad>(baseOp);

            int inputLength = untypedInput.Shape.Size;
            Shape inputShape = untypedInput.Shape;
            Shape outputShape = untypedOutput.Shape;

            ParallelFor(0, inputLength, range =>
            {
                Span<TType> input = untypedInput.GetInitializedDataSpan<TType>();
                Span<TGrad> gradInput = untypedInput.GetInitializedGrad<TGrad>();
                Span<TType> output = untypedOutput.GetInitializedDataSpan<TType>();
                Span<TGrad> gradOutput = untypedOutput.GetInitializedGrad<TGrad>();

                for (int iInput = range.Item1; iInput < range.Item2; iInput++)
                {
                    int iOutput = outputShape.GetLinearIndex(iInput, inputShape, false);

                    TType complement = inverseOperation(output[iOutput], input[iInput]);
                    TGrad grad = backwardRight(complement, input[iInput], gradOutput[iOutput]);
                    
                    gradInput[iInput] += grad;
                }
            });
        }
    }
}

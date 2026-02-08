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
        /// Executes the forward pass of a reduction operation for a given input and stores the result in the given output.
        /// </summary>
        /// <typeparam name="TType">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <param name="kind">The kind of reduction operation to execute.</param>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <param name="reduceDim">The dimension to reduce over.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{T}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="BinaryOperations"/> forward method is not found.</exception>
        private void ExecuteReductionForward<TType>(KindReduction kind, Value untypedInput, Value untypedOutput, Dimension reduceDim)
            where TType : struct, INumber<TType>
        {
            TType[] input = untypedInput.GetInitializedData<TType>();
            TType[] output = untypedOutput.GetOrInitializeData<TType>();

            // Extract the base binary operation from the reduction kind
            KindBinary baseOp = kind.GetBaseOperation();

            // Get the appropriate forward method for the base binary operation
            Func<TType, TType, TType> operation = BinaryOperations.GetKindForwardDelegate<TType>(baseOp);

            //// Initialize output with neutral element
            FillArray(output, untypedInput.Kind.GetNeutralElement<TType>());

            // Reduce the single dimension
            Reduce(input, untypedInput.Shape, output, reduceDim, operation);
        }

        public static void FillArray<TType>(TType[] output, TType neutralElement)
            where TType : struct, INumber<TType>
        {
            int iOutput = 0;
            if (Vector<TType>.IsSupported && output.Length >= Vector<TType>.Count)
            {
                Span<TType> outputSpan = output.AsSpan();
                Vector<TType> neutralVector = new(neutralElement);
                int iVectorEnd = output.Length - (output.Length % Vector<TType>.Count);
                for (; iOutput < iVectorEnd; iOutput += Vector<TType>.Count)
                {
                    neutralVector.CopyTo(outputSpan[iOutput..]);
                }
            }
            for (; iOutput < output.Length; iOutput++)
            {
                output[iOutput] = neutralElement;
            }
        }

        internal void Reduce<TType>(TType[] input, Shape inputShape, TType[] output, Dimension reduceDim, Func<TType, TType, TType> operation)
         where TType : struct, INumber<TType>
        {
            Shape outputShape = inputShape.Remove(reduceDim);
            // Find the dimension index in input shape
            int dimIndex = inputShape.IndexOf(reduceDim);
            if (dimIndex < 0)
            {
                throw new ArgumentException($"Dimension {reduceDim} not found in input shape {inputShape}.");
            }

            int dimSize = reduceDim.Size;
            int dimStride = inputShape.GetStride(dimIndex);
            int outputLength = output.Length;

            // For each position in output, accumulate along the dimension to reduce
            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = 0; iOutput < outputLength; iOutput++)
                {
                    // Map output index to input base index using Shape.GetLinearIndex
                    int iInputBase = inputShape.GetLinearIndex(iOutput, outputShape);

                    // Accumulate along the dimension to reduce
                    TType accumulator = output[iOutput];
                    int iInputEnd = iInputBase + dimSize * dimStride;
                    for (int iInput = iInputBase; iInput < iInputEnd; iInput += dimStride)
                    {
                        accumulator = operation(accumulator, input[iInput]);
                    }
                    output[iOutput] = accumulator;
                }
            }
            else
            {
                Parallel.For(0, outputLength, _parallelOptions, iOutput =>
                {
                    // Map output index to input base index using Shape.GetLinearIndex
                    int iInputBase = inputShape.GetLinearIndex(iOutput, outputShape);

                    // Accumulate along the dimension to reduce
                    TType accumulator = output[iOutput];
                    int iInputEnd = iInputBase + dimSize * dimStride;
                    for (int iInput = iInputBase; iInput < iInputEnd; iInput += dimStride)
                    {
                        accumulator = operation(accumulator, input[iInput]);
                    }
                    output[iOutput] = accumulator;
                });
            }
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
        /// <typeparam name="T">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="G">The type of the gradients. Must be a struct that implements <see cref="IFloatingPointIeee754{G}"/>.</typeparam>
        /// <param name="kind">The kind of reduction operation.</param>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <param name="reduceDim">The dimension that was reduced.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the backward method is not found.</exception>
        private void ExecuteReductionBackward<T, G>(KindReduction kind, Value untypedInput, Value untypedOutput, Dimension reduceDim)
            where T : struct, INumber<T>
            where G : struct, IFloatingPointIeee754<G>
        {
            T[] input = untypedInput.GetInitializedData<T>();
            G[] gradInput = untypedInput.GetOrInitializeGrad<G>();
            T[] output = untypedOutput.GetInitializedData<T>();
            G[] gradOutput = untypedOutput.GetInitializedGrad<G>();

            // Extract the base binary operation from the reduction kind
            KindBinary baseOp = (KindBinary)((int)kind & ~(int)KindCategory.Reduction);
            
            // Get the inverse operation
            KindBinary inverseOp = (KindBinary)((int)baseOp | (int)KindProperty.Inverse);

            // Get the inverse operation forward method
            Func<T, T, T> inverseOperation = BinaryOperations.GetKindForwardDelegate<T>(inverseOp);

            // Get the backward right method for the base operation
            Func<T, T, G, G> backwardRight = BinaryOperations.GetKindBackwardRightDelegate<T, G>(baseOp);

            // Propagate gradients
            int inputLength = input.Length;
            Shape inputShape = untypedInput.Shape;
            Shape outputShape = untypedOutput.Shape;

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iInput = inputLength - 1; iInput >= 0; iInput--)
                {
                    // Map input index to output index
                    int iOutput = outputShape.GetLinearIndex(iInput, inputShape);

                    // Calculate the complement: the reduced value without the current input element
                    // complement = inverseOperation(output, input[i])
                    T complement = inverseOperation(output[iOutput], input[iInput]);
                    
                    // Calculate the gradient using the backward right operation
                    // grad = backwardRight(complement, input[i], gradOutput)
                    G grad = backwardRight(complement, input[iInput], gradOutput[iOutput]);
                    
                    // Accumulate the gradient
                    gradInput[iInput] += grad;
                }
            }
            else
            {
                Parallel.For(0, inputLength, _parallelOptions, iInput =>
                {
                    // Map input index to output index
                    int iOutput = outputShape.GetLinearIndex(iInput, inputShape);

                    // Calculate the complement: the reduced value without the current input element
                    // complement = inverseOperation(output, input[i])
                    T complement = inverseOperation(output[iOutput], input[iInput]);
                    
                    // Calculate the gradient using the backward right operation
                    // grad = backwardRight(complement, input[i], gradOutput)
                    G grad = backwardRight(complement, input[iInput], gradOutput[iOutput]);
                    
                    // Accumulate the gradient
                    gradInput[iInput] += grad;
                });
            }
        }
    }
}

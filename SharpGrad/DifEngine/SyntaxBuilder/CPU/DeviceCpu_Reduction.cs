//#define MP
using SharpGrad.DifEngine.SyntaxBuilder.Operations;
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
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteReductionForward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(KindReduction), typeof(Value), typeof(Value), typeof(Dimension)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteReductionForward)} not found.");
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
            // Throw an exception if the input and output are not of the expected type
            if (untypedInput is not Value<TType> inputValue)
            {
                throw new InvalidCastException($"'{nameof(untypedInput)}' must be {nameof(Value)}<{nameof(TType)}>.");
            }
            if (untypedOutput is not Value<TType> outputValue)
            {
                throw new InvalidCastException($"'{nameof(untypedOutput)}' must be {nameof(Value)}<{nameof(TType)}>.");
            }

            DataBuffer<TType> inputData = inputValue.data;
            ThrowIfNotInitialized(inputData);
            DataBuffer<TType> outputData = outputValue.data;
            outputData.Initialize();

            TType[] input = inputData.flatData!;
            TType[] output = outputData.flatData!;

            // Extract the base binary operation from the reduction kind
            KindBinary baseOp = kind.GetBaseOperation();

            // Get the appropriate forward method for the base binary operation
            if (!cacheBinaryKindForwardMethodInfos.TryGetValue((baseOp, typeof(TType)), out MethodInfo? method))
            {
                string methodName = $"{baseOp}Forward";
                method = typeof(BinaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(TType), typeof(TType)]
                ) ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                method = method.MakeGenericMethod(typeof(TType));
                cacheBinaryKindForwardMethodInfos[(baseOp, typeof(TType))] = method;
            }
            Func<TType, TType, TType> operation = method.CreateDelegate<Func<TType, TType, TType>>();

            // Get the neutral element for the operation
            TType neutralElement = outputValue.Kind.GetNeutralElement<TType>();

            // Initialize output with neutral element
            int outputLength = output.Length;
            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = 0; iOutput < outputLength; iOutput++)
                {
                    output[iOutput] = neutralElement;
                }
            }
            else
            {
                Parallel.For(0, outputLength, _parallelOptions, iOutput =>
                {
                    output[iOutput] = neutralElement;
                });
            }

            // Reduce the single dimension
            Reduce(input, inputValue.Shape, output, reduceDim, operation);
        }

        internal void InitializeOutputForReduction<TType>(TType[] output, KindGraphNode kind)
            where TType : struct, INumber<TType>
        {
            TType neutralElement = kind.GetNeutralElement<TType>();
            int outputLength = output.Length;
            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = 0; iOutput < outputLength; iOutput++)
                {
                    output[iOutput] = neutralElement;
                }
            }
            else
            {
                Parallel.For(0, outputLength, _parallelOptions, iOutput =>
                {
                    output[iOutput] = neutralElement;
                });
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
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteReductionBackward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(KindReduction), typeof(Value), typeof(Value), typeof(Dimension)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteReductionBackward)} not found.");
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
            if (untypedInput is not Value<T> inputValue)
            {
                throw new InvalidCastException($"'{nameof(untypedInput)}' must be {nameof(Value)}<{nameof(T)}>.");
            }
            if (untypedOutput is not Value<T> outputValue)
            {
                throw new InvalidCastException($"'{nameof(untypedOutput)}' must be {nameof(Value)}<{nameof(T)}>.");
            }

            DataBuffer<T> inputData = inputValue.data;
            ThrowIfNotInitialized(inputData);
            DataBuffer<G> inputGrad = inputValue.InitializeGrad<G>();
            DataBuffer<T> outputData = outputValue.data;
            ThrowIfNotInitialized(outputData);
            DataBuffer<G> outputGrad = GetInitializedBuffer<G>(outputValue.untypedGrad);

            T[] input = inputData.flatData!;
            G[] gradInput = inputGrad.flatData!;
            T[] output = outputData.flatData!;
            G[] gradOutput = outputGrad.flatData!;

            // Extract the base binary operation from the reduction kind
            KindBinary baseOp = (KindBinary)((int)kind & ~(int)KindCategory.Reduction);
            
            // Get the inverse operation
            KindBinary inverseOp = (KindBinary)((int)baseOp | (int)KindProperty.Inverse);

            // Get the inverse operation forward method
            if (!cacheBinaryKindForwardMethodInfos.TryGetValue((inverseOp, typeof(T)), out MethodInfo? inverseMethod))
            {
                string methodName = $"{inverseOp}Forward";
                inverseMethod = typeof(BinaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(T), typeof(T)]
                ) ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                inverseMethod = inverseMethod.MakeGenericMethod(typeof(T));
                cacheBinaryKindForwardMethodInfos[(inverseOp, typeof(T))] = inverseMethod;
            }
            Func<T, T, T> inverseOperation = inverseMethod.CreateDelegate<Func<T, T, T>>();

            // Get the backward right method for the base operation
            if (!cacheBinaryKindBackwardRightMethodInfos.TryGetValue((baseOp, typeof(T), typeof(G)), out MethodInfo? backwardMethod))
            {
                string methodName = $"{baseOp}BackwardRight";
                backwardMethod = typeof(BinaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(T), typeof(T), typeof(G)]
                ) ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                backwardMethod = backwardMethod.MakeGenericMethod(typeof(T), typeof(G));
                cacheBinaryKindBackwardRightMethodInfos[(baseOp, typeof(T), typeof(G))] = backwardMethod;
            }
            Func<T, T, G, G> backwardRight = backwardMethod.CreateDelegate<Func<T, T, G, G>>();

            // Propagate gradients
            int inputLength = input.Length;
            Shape inputShape = inputValue.Shape;
            Shape outputShape = outputValue.Shape;

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iInput = 0; iInput < inputLength; iInput++)
                {
                    // Find the corresponding output index
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
                    // Find the corresponding output index
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

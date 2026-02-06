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
        // Cache for the ExecuteForward MethodInfos
        private static readonly Dictionary<
            (Type Type, KindGraphNode kind),
            MethodInfo> cacheUnaryKindForwardMethodInfos = [];

        // Cache for the ExecuteUnaryForward delegates
        private readonly Dictionary<
            Type,
            Action<Value, Value>> cacheExecuteUnaryForwards = [];

        /// <summary>
        /// Executes the forward pass of a unary operation on the given input and stores the result in the given output.
        /// </summary>
        /// <param name="input">The input Value.</param>
        /// <param name="output">The output Value.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the forward pass for different element types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteUnaryForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteUnaryForward(Value input, Value output)
        {
            // Check if a cached delegate exists for the input element type
            Type inputElement = input.ElementType;
            if (!cacheExecuteUnaryForwards.TryGetValue(inputElement, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteUnaryForward<T>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteUnaryForward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteUnaryForward)} not found.");
                method = method.MakeGenericMethod(inputElement);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteUnaryForwards[inputElement] = action;
            }
            // Call the generic ExecuteUnaryForward<T> method
            action(input, output);
        }

        /// <summary>
        /// Executes the forward pass of a unary operation for a given input and stores the result in the given output.
        /// </summary>
        /// <typeparam name="TType">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{T}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="UnaryOperations"/>."[<see cref="Value.Kind"/>]Forward" method is not found.</exception>
        private void ExecuteUnaryForward<TType>(Value untypedInput, Value untypedOutput)
            where TType : struct, INumber<TType>
        {
            // Throw an exception if the input and output are not of the expected type
            if (untypedInput is not Value<TType> inputValue)
            {
                throw new InvalidCastException($"'{nameof(untypedInput)}' must be {nameof(Value)}<{nameof(TType)}>.");
            }
            // Throw an exception if the input and output are not of the expected type
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

            // Get the appropriate method for the unary operation
            if (!cacheUnaryKindForwardMethodInfos.TryGetValue((typeof(TType), untypedOutput.Kind), out MethodInfo? method))
            {
                string methodName = $"{untypedOutput.Kind}Forward";
                method = typeof(UnaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(TType), typeof(TType)]
                ) ?? throw new InvalidOperationException($"Method {nameof(UnaryOperations)}.{methodName} not found.");
                method = method.MakeGenericMethod(typeof(TType));
                cacheUnaryKindForwardMethodInfos[(typeof(TType), untypedOutput.Kind)] = method;
            }
            Func<TType, TType> operation = method.CreateDelegate<Func<TType, TType>>();

            // Perform the unary operation
            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = input.Length - 1; iOutput >= 0; iOutput--)
                {
                    output[iOutput] = operation(input[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, input.Length, _parallelOptions, iOutput =>
                {
                    output[iOutput] = operation(input[iOutput]);
                });
            }
        }

        // Cache for the ExecuteCastForward delegates
        private readonly Dictionary<
            (Type From, Type To),
            Action<Value, Value>> cacheExecuteCastForwards = [];

        /// <summary>
        /// Executes the forward pass of a cast operation from input to output.
        /// </summary>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the cast operation for different combinations of source and target types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteCastForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteCastForward(Value input, Value output)
        {
            // Check if a cached delegate exists for the input and output element types
            (Type From, Type To) key = (input.ElementType, output.ElementType);
            if (!cacheExecuteCastForwards.TryGetValue(key, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteCastForward<TFrom, TTo>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteCastForward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteCastForward)} not found.");
                method = method.MakeGenericMethod(key.From, key.To);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteCastForwards[key] = action;
            }
            // Call the generic CastForward<TFrom, TTo> method
            action(input, output);
        }

        /// <summary>
        /// Executes the forward pass of a cast operation from type TFrom to type TTo.
        /// </summary>
        /// <typeparam name="TFrom">The source type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TTo">The target type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> must be of type <see cref="Value{TFrom}"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{TTo}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input and output are not of the expected types.</exception>
        private void ExecuteCastForward<TFrom, TTo>(Value untypedInput, Value untypedOutput)
            where TFrom : struct, INumber<TFrom>
            where TTo : struct, INumber<TTo>
        {
            if (untypedInput is not Value<TFrom> inputValue)
            {
                throw new InvalidCastException($"Input must be {nameof(Value)}<{nameof(TFrom)}>.");
            }
            if (untypedOutput is not Value<TTo> outputValue)
            {
                throw new InvalidCastException($"Output must be {nameof(Value)}<{nameof(TTo)}>.");
            }

            DataBuffer<TFrom> inputData = inputValue.data;
            ThrowIfNotInitialized(inputData);
            DataBuffer<TTo> outputData = outputValue.data;
            outputData.Initialize();

            TFrom[] input = inputData.flatData!;
            TTo[] output = outputData.flatData!;

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = output.Length - 1; iOutput >= 0; iOutput--)
                {
                    output[iOutput] = UnaryOperations.CastForward<TFrom, TTo>(input[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, output.Length, _parallelOptions, iOutput =>
                {
                    output[iOutput] = UnaryOperations.CastForward<TFrom, TTo>(input[iOutput]);
                });
            }
        }

        // Cache for the ExecuteBackward delegates MethodInfos
        private static readonly Dictionary<
            (KindGraphNode kind, Type Value, Type Gradient),
            MethodInfo> cacheExecuteBackwardMethodsInfos = [];

        // Cache for the ExecuteUnaryBackward delegates
        private readonly Dictionary<
            (Type Value, Type Gradient),
            Action<Value, Value>> cacheExecuteUnaryBackwards = [];

        /// <summary>
        /// Executes the backward pass of a unary operation on the given input and output Values.
        /// </summary>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the ExecuteUnaryBackward method is not found or if the input and output types are incompatible.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteUnaryBackward(Value input, Value output)
        {
            // Call the generic ExecuteUnaryBackward<T, G>(Value, Value) method
            (Type Value, Type Gradient) key = (output.ElementType, output.Grad.ElementType);
            if (!cacheExecuteUnaryBackwards.TryGetValue(key, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteUnaryBackward<T, G>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteUnaryBackward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    new[] { typeof(Value), typeof(Value) }
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteUnaryBackward)} not found.");
                method = method.MakeGenericMethod(key.Value, key.Gradient);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteUnaryBackwards[key] = action;
            }
            action(input, output);
        }

        /// <summary>
        /// Executes the backward pass of a unary operation on the given input and output Values.
        /// </summary>
        /// <typeparam name="T">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="G">The type of the gradients. Must be a struct that implements <see cref="IFloatingPointIeee754{G}"/>.</typeparam>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="UnaryOperations"/>."[<see cref="Value.Kind"/>]Backward" method is not found.</exception>
        private void ExecuteUnaryBackward<T, G>(Value untypedInput, Value untypedOutput)
            where T : struct, INumber<T>
            where G : struct, IFloatingPointIeee754<G>
        {
            if (untypedInput is not Value<T> inputValue)
            {
                // Throw an exception if the input and output are not of the expected type
                throw new InvalidCastException($"'{nameof(untypedInput)}' must be {nameof(Value)}<{nameof(T)}>.");
            }
            if (untypedOutput is not Value<T> outputValue)
            {
                // Throw an exception if the input and output are not of the expected type
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

            // Get the appropriate method for the unary operation
            if (!cacheExecuteBackwardMethodsInfos.TryGetValue((outputValue.Kind, typeof(T), typeof(G)), out MethodInfo? method))
            {
                string methodName = $"{outputValue.Kind}Backward";
                method = typeof(UnaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(T), typeof(T), typeof(G)]
                ) ?? throw new InvalidOperationException($"Method {nameof(UnaryOperations)}.{methodName} not found.");
                method = method.MakeGenericMethod(typeof(T), typeof(G));
                cacheExecuteBackwardMethodsInfos[(outputValue.Kind, typeof(T), typeof(G))] = method;
            }
            // Create a delegate for the method
            Func<T, T, G, G> func = method.CreateDelegate<Func<T, T, G, G>>();

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = gradOutput.Length - 1; iOutput >= 0; iOutput--)
                {
                    gradInput[iOutput] += func(input[iOutput], output[iOutput], gradOutput[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, gradOutput.Length, _parallelOptions, iOutput =>
                {
                    gradInput[iOutput] += func(input[iOutput], output[iOutput], gradOutput[iOutput]);
                });
            }
        }

        // Cache for the ExecuteCastBackward delegates
        private readonly Dictionary<
            (Type From, Type To, Type Grad),
            Action<Value, Value>> cacheExecuteCastBackwards = [];

        /// <summary>
        /// Executes the backward pass of a cast operation from output to input.
        /// </summary>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the cast operation for different combinations of source and target types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteCastBackward"/> method is not found.</exception>
        private void ExecuteCastBackward(Value input, Value output)
        {
            (Type From, Type To, Type Grad) key = (input.ElementType, output.ElementType, output.Grad.ElementType);
            if (!cacheExecuteCastBackwards.TryGetValue(key, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteCastBackward<From, To, Grad>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteCastBackward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteCastBackward)} not found.");
                method = method.MakeGenericMethod(key.From, key.To, key.Grad);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate< Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteCastBackwards[key] = action;
            }
            // Call the generic CastBackward<TFrom, TTo, G> method
            action(input, output);
        }

        /// <summary>
        /// Executes the backward pass of a cast operation from type TTo to type TFrom.
        /// </summary>
        /// <typeparam name="TFrom">The source type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TTo">The target type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="G">The type of the gradients. Must be a struct that implements <see cref="IFloatingPointIeee754{G}"/>.</typeparam>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> must be of type <see cref="Value{TFrom}"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{TTo}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input and output are not of the expected types.</exception>
        private void ExecuteCastBackward<TFrom, TTo, G>(Value untypedInput, Value untypedOutput)
            where TFrom : struct, INumber<TFrom>
            where TTo : struct, INumber<TTo>
            where G : struct, IFloatingPointIeee754<G>
        {
            if (!untypedInput.IsGradiable)
                return;

            if (untypedInput is not Value<TFrom> input)
            {
                throw new InvalidCastException($"Input must be {nameof(Value)}<{nameof(TFrom)}>.");
            }
            if (untypedOutput is not Value<TTo> output)
            {
                throw new InvalidCastException($"Output must be {nameof(Value)}<{nameof(TTo)}>.");
            }

            ThrowIfNotInitialized(input.data);
            DataBuffer<G> typedGradInput = input.InitializeGrad<G>();
            ThrowIfNotInitialized(output.data);
            DataBuffer<G> typedGradOutput = GetInitializedBuffer<G>(output.untypedGrad);

            TFrom[] inputValue = input.data.flatData!;
            TTo[] outputValue = output.data.flatData!;
            G[] gradOutputValue = typedGradOutput.flatData!;
            G[] gradInputValue = typedGradInput.flatData!;

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = inputValue.Length - 1; iOutput >= 0; iOutput--)
                {
                    gradInputValue[iOutput] += UnaryOperations.CastBackward(inputValue[iOutput], outputValue[iOutput], gradOutputValue[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, inputValue.Length, _parallelOptions, iOutput =>
                {
                    gradInputValue[iOutput] += UnaryOperations.CastBackward(inputValue[iOutput], outputValue[iOutput], gradOutputValue[iOutput]);
                });
            }
        }
    }
}
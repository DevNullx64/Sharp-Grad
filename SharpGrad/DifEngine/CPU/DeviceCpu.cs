using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public partial class DeviceCpu
    {
        ParallelOptions _parallelOptions;
        public int ThreadCount
        {
            get => _parallelOptions.MaxDegreeOfParallelism;
            set => _parallelOptions.MaxDegreeOfParallelism = value;
        }

        public DeviceCpu(int cpuCount)
        {
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = cpuCount };
        }
        public DeviceCpu() :
            this(Environment.ProcessorCount)
        { }


        public bool IsAvailable() => true;

        public string Name => nameof(DeviceCpu);

        public void ResetGradient<TType>(Value<TType> loss)
            where TType : struct, INumber<TType>
        {
            loss.untypedGrad = null;
        }

        /// <summary>
        /// Helper method to find a generic method by name and signature.
        /// </summary>
        /// <param name="methodName">The name of the method to find.</param>
        /// <param name="genericArgCount">The number of generic type parameters.</param>
        /// <param name="parameterTypes">The parameter types of the method (excluding generic parameters).</param>
        /// <returns>The generic MethodInfo, or null if not found.</returns>
        private static MethodInfo? GetGenericMethod(string methodName, int genericArgCount, params Type[] parameterTypes)
        {
            return typeof(DeviceCpu).GetMethods(BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == methodName &&
                    m.IsGenericMethodDefinition &&
                    m.GetGenericArguments().Length == genericArgCount &&
                    m.GetParameters().Length == parameterTypes.Length &&
                    m.GetParameters().Select(p => p.ParameterType).SequenceEqual(parameterTypes));
        }

        /// <summary>
        /// Helper method to find a generic method definition by name from a specific type.
        /// </summary>
        /// <param name="type">The type to search for the method.</param>
        /// <param name="methodName">The name of the method to find.</param>
        /// <param name="bindingFlags">The binding flags for the search.</param>
        /// <returns>The generic MethodInfo, or null if not found.</returns>
        internal static MethodInfo? FindGenericMethod(Type type, string methodName, BindingFlags bindingFlags = BindingFlags.Public | BindingFlags.Static)
        {
            var methods = type.GetMethods(bindingFlags);
            foreach (var m in methods)
            {
                if (m.Name == methodName && m.IsGenericMethodDefinition)
                {
                    return m;
                }
            }
            return null;
        }
    }
}

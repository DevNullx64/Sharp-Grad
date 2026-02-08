using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.CPU
{
    public partial class DLLCompiler
    {
/*
        private static Dictionary<string, MethodDeclarationSyntax> _generatedUnaryMethods = [];
        public static string BuildUnarySourceCodeForCPU(UnaryMethodInfo indicesInfo, string @namespace, string className)
        {
            if (_generatedUnaryMethods.ContainsKey(indicesInfo.MethodSuffix))
            {
                return string.Empty; // Method already generated
            }
            string parametersIndiceNames = string.Join(", ", indicesInfo.InputIndices.Select(name => $"int {name}"));
            string inputIndiceNames = string.Join(", ", indicesInfo.InputIndices);
            string arrayIndices = new(',', indicesInfo.InputIndices.Count - 1);
            return $@"
using System;
using {typeof(INumber<>).Namespace!};
using {typeof(IOperation<>).Namespace!};
using {typeof(MethodImplAttribute).Namespace!};

namespace {@namespace}
{{
    public static partial class {className}
    {{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Op1_{indicesInfo.MethodSuffix}<O, T>({parametersIndiceNames}, T[{arrayIndices}] input, T[{arrayIndices}] result)
            where O : {typeof(IUnaryOperation<>).GetName("T")}
            where T : {typeof(INumber<>).GetName("T")}
        {{
            result[{inputIndiceNames}] = O.{nameof(IUnaryOperation<float>.Operate)}(input[{inputIndiceNames}]);
        }}
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Op1Backwards_{indicesInfo.MethodSuffix}<O, T, G>({parametersIndiceNames}, T[{arrayIndices}] input, G[{arrayIndices}] inputGrad, T[{arrayIndices}] result, G[{arrayIndices}] resultGrad)
            where O : {typeof(IUnaryOperation<>).GetName("T")}
            where T : {typeof(INumber<>).GetName("T")}
            where G : {typeof(IFloatingPoint<>).GetName("G")}
        {{
            inputGrad[{inputIndiceNames}] += O.{nameof(IUnaryOperation<float>.Backward)}(input[{inputIndiceNames}], resultGrad[{inputIndiceNames}]);
        }}
    }}
}}";
        }
        public static string BuildUnarySourceCodeForCPU(Shape input, string @namespace, string className)
        {
            UnaryMethodInfo indicesInfo = UnaryMethodInfo.Create(input);
            return BuildUnarySourceCodeForCPU(indicesInfo, @namespace, className);
        }
        public static SyntaxTree GetUnarySyntaxTreeForCPU(Shape input, string @namespace, string className)
        {
            string sourceCode = BuildUnarySourceCodeForCPU(input, @namespace, className);
            return CSharpSyntaxTree.ParseText(sourceCode);
        }

        private static Dictionary<string, MethodDeclarationSyntax> _generatedBinaryMethods = [];
        public static string BuildBinarySourceCodeForCPU(BinaryMethodInfo indicesInfo, string @namespace, string className)
        {
            if(_generatedBinaryMethods.ContainsKey(indicesInfo.MethodSuffix))
            {
                return string.Empty; // Method already generated
            }
            string parametersIndiceNames = string.Join(", ", indicesInfo.ResultIndices.Select(name => $"int {name}"));
            string leftIndiceNames = string.Join(", ", indicesInfo.LeftIndices);
            string leftArrayIndices = new(',', indicesInfo.LeftIndices.Count - 1);
            string rightIndiceNames = string.Join(", ", indicesInfo.RightIndices);
            string rightArrayIndices = new(',', indicesInfo.RightIndices.Count - 1);
            string resultIndiceNames = string.Join(", ", indicesInfo.ResultIndices);
            string resultArrayIndices = new(',', indicesInfo.ResultIndices.Count - 1);

            return $@"
using System;
using {typeof(INumber<>).Namespace!};
using {typeof(IOperation<>).Namespace!};
using {typeof(MethodImplAttribute).Namespace!};

namespace {@namespace}
{{
    public static partial class {className}
    {{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Op2_{indicesInfo.MethodSuffix}<O, T>({parametersIndiceNames}, T[{leftArrayIndices}] left, T[{rightArrayIndices}] right, T[{resultArrayIndices}] result)
            where O : {typeof(IBinaryOperation<>).GetName("T")}
            where T : {typeof(INumber<>).GetName("T")}
        {{
            result[{resultIndiceNames}] = O.{nameof(IBinaryOperation<float>.Operate)}(left[{leftIndiceNames}], right[{rightIndiceNames}]);
        }}
    
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Op2Backwards_{indicesInfo.MethodSuffix}<O, T, G>({parametersIndiceNames}, T[{leftArrayIndices}] left, G[{leftArrayIndices}] leftGrad, T[{rightArrayIndices}] right, G[{rightArrayIndices}] rightGrad, T[{resultArrayIndices}] result, G[{resultArrayIndices}] resultGrad)
            where O : {typeof(IBinaryOperation<>).GetName("T")}
            where T : {typeof(INumber<>).GetName("T")}
            where G : {typeof(IFloatingPoint<>).GetName("G")}
        {{
            var gradients = O.{nameof(IBinaryOperation<float>.Backward)}(left[{leftIndiceNames}], right[{rightIndiceNames}], resultGrad[{resultIndiceNames}]);
            leftGrad[{leftIndiceNames}] += gradients.Left;
            rightGrad[{rightIndiceNames}] += gradients.Right;
        }}
    }}
}}";
        }
        public static string BuildBinarySourceCodeForCPU(Shape left, Shape right, string @namespace, string className)
        {
            BinaryMethodInfo indicesInfo = BinaryMethodInfo.Create(left, right);
            return BuildBinarySourceCodeForCPU(indicesInfo, @namespace, className);
        }
        public static SyntaxTree GetBinarySyntaxTreeForCPU(Shape left, Shape right, string @namespace, string className)
        {
            string sourceCode = BuildBinarySourceCodeForCPU(left, right, @namespace, className);
            return CSharpSyntaxTree.ParseText(sourceCode);
        }

        public static Dictionary<string, MethodDeclarationSyntax> _generatedReducMethods = [];
        public static string BuildTileReductionSourceCodeForCPU(Shape input, int reduceDimension, string @namespace, string className)
        {
            ReductionMethodInfo indicesInfo = ReductionMethodInfo.Create(input, reduceDimension);
            if (_generatedReducMethods.ContainsKey(indicesInfo.MethodSuffix))
            {
                return string.Empty;
            }
            
            string parametersIndiceNames = string.Join(", ", indicesInfo.InputIndices.Select(name => $"int {name}"));
            string arrayIndices = new(',', indicesInfo.InputIndices.Count - 1);
            string reducedIndiceNames = string.Join(", ", indicesInfo.InputIndices.Where((_, d) => d != indicesInfo.ReduceDimension));
            string reducedIndices = new(',', indicesInfo.InputIndices.Count - 2);
            string reduceIndice = indicesInfo.InputIndices[indicesInfo.ReduceDimension];
            string idxVariable = indicesInfo.InputIndices[indicesInfo.ReduceDimension];

            // Build the input access indices - replace the reduced dimension with "idx"
            List<string> inputAccessIndices = new(indicesInfo.InputIndices.Count);
            for (int d = 0; d < indicesInfo.InputIndices.Count; d++)
            {
                if (d == indicesInfo.ReduceDimension)
                {
                    inputAccessIndices.Add("idx");
                }
                else
                {
                    inputAccessIndices.Add(indicesInfo.InputIndices[d]);
                }
            }
            string inputAccessString = string.Join(", ", inputAccessIndices);
            
            // Result access uses all parameter indices
            string resultAccessString = string.Join(", ", indicesInfo.InputIndices);
            
            // Name of the tile index variable (e.g., "i2" for reduceDimension=2)
            string tileIndexName = indicesInfo.InputIndices[indicesInfo.ReduceDimension];
            return $@"
using System;
using {typeof(INumber<>).Namespace!};
using {typeof(IOperation<>).Namespace!};
using {typeof(MethodImplAttribute).Namespace!};

namespace {@namespace}
{{
    public static partial class {className}
    {{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TileReduce_{indicesInfo.MethodSuffix}<O, T>({parametersIndiceNames}, T[{arrayIndices}] input, T[{arrayIndices}] result)
            where O : {typeof(IReductionOperation<>).GetName("T")}
            where T : {typeof(INumber<>).GetName("T")}
        {{
            int inputSize = input.GetLength({indicesInfo.ReduceDimension});
            int resultSize = result.GetLength({indicesInfo.ReduceDimension});
            int tileSize = (int)MathF.Ceiling((float)inputSize / resultSize);
            
            T accumulator = O.InitialValue;
            int startIdx = {tileIndexName} * tileSize;
            int stopIdx = startIdx + tileSize;
            if (stopIdx > inputSize) stopIdx = inputSize;
            
            for (int idx = startIdx; idx < stopIdx; idx++)
            {{
                accumulator = O.{nameof(IReductionOperation<float>.Reduce)}(accumulator, input[{inputAccessString}]);
            }}
            
            result[{resultAccessString}] = accumulator;
        }}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Reduce_{indicesInfo.MethodSuffix}<O, T>({reducedIndiceNames}, T[{arrayIndices}] input, T[{reducedIndices}] result)
            where O : {typeof(IReductionOperation<>).GetName("T")}
            where T : {typeof(INumber<>).GetName("T")}
        {{
            int inputSize = input.GetLength({indicesInfo.ReduceDimension});
            T accumulator = O.InitialValue;
            
            for (int {idxVariable} = 0; {idxVariable} < inputSize; {idxVariable}++)
            {{
                accumulator = O.{nameof(IReductionOperation<float>.Reduce)}(accumulator, input[{inputAccessString}]);
            }}
            
            result[{reducedIndiceNames}] = accumulator;
        }}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReduceBackwards_{indicesInfo.MethodSuffix}<G>({reducedIndiceNames}, G[{arrayIndices}] inputGrad, G[{reducedIndices}] resultGrad)
            where G : {typeof(IFloatingPoint<>).GetName("G")}
        {{
            int inputSize = inputGrad.GetLength({indicesInfo.ReduceDimension});
            G resultGradValue = resultGrad[{reducedIndiceNames}];
            
            for (int {idxVariable} = 0; {idxVariable} < inputSize; {idxVariable}++)
            {{
                inputGrad[{inputAccessString}] += resultGradValue;
            }}
        }}
    }}
}}";
        }

        public static PortableExecutableReference GetMetadataReference(Type type)
        {
            string? assemblyLocation = type.Assembly.Location;
            if (!string.IsNullOrEmpty(assemblyLocation) && File.Exists(assemblyLocation))
            {
                return MetadataReference.CreateFromFile(assemblyLocation);
            }
            throw new FileNotFoundException($"Assembly location for type '{type.FullName}' could not be found.");
        }

        public static IEnumerable<PortableExecutableReference> GetMetadataReferences(params Type[] coreAssemblyTypes)
        {
            List<PortableExecutableReference> references = [
                GetMetadataReference(typeof(object)),
                GetMetadataReference(typeof(AssemblyTargetedPatchBandAttribute))
            ];

            // 1. Explicitly add core assemblies by location if available
            foreach (Type type in coreAssemblyTypes)
            {
                PortableExecutableReference? reference = GetMetadataReference(type);
                if (reference != null)
                {
                    references.Add(reference);
                }
            }

            // 2. Add TRUSTED_PLATFORM_ASSEMBLIES to cover everything else
            string? trustedAssemblies = (string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES");
            if (trustedAssemblies != null)
            {
                foreach (var path in trustedAssemblies.Split(Path.PathSeparator))
                {
                    references.Add(MetadataReference.CreateFromFile(path));
                }
            }
            // 3. Return deduplicated by path
            return references
                .GroupBy(r => r.FilePath)
                .Select(g => g.First());
        }
*/
    }
}
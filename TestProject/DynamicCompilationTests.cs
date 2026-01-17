using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpGrad;
using SharpGrad.SyntaxBuilder;
using System.Numerics;
using System.Reflection;
using System.Runtime;

namespace TestProject.Compilation
{
    [TestClass]
    public class DynamicCompilationTests
    {
        [TestMethod]
        public void BuildAssemblySample_Compile()
        {
            string namespaceName = "DynamicNamespace";
            string className = "Test";

            Dimension x = new("x", 3);
            Dimension y = new("y", 4);
            Dimension z = new("z", 5);
            Shape leftShape = (x, y);
            Shape rightShape = (y, z);

            // 1. Get the syntax tree
            SyntaxTree syntaxOp1Tree = DLLCompiler.GetUnarySyntaxTreeForCPU(leftShape, namespaceName, className);
            SyntaxTree syntaxOp2Tree = DLLCompiler.GetBinarySyntaxTreeForCPU(leftShape, rightShape, namespaceName, className);

            // 2. Necessary references
            var references = DLLCompiler.GetMetadataReferences(
                typeof(object), // mscorlib
                typeof(AssemblyTargetedPatchBandAttribute), // System.Runtime
                typeof(INumber<>), // System.Numerics
                typeof(IOperation<>) // SharpGrad.SyntaxBuilder
            );

            // 3. In-memory compilation
            CSharpCompilation compilation = CSharpCompilation.Create(
                namespaceName,
                [syntaxOp1Tree, syntaxOp2Tree],
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary)
            );

            using MemoryStream ms = new();
            EmitResult result = compilation.Emit(ms);

            if (!result.Success)
            {
                throw new Exception("Compilation failed:\n" + string.Join("\n", result.Diagnostics.Select(d => d.ToString())));
            }

            // 4. Loading the assembly into memory
            ms.Seek(0, SeekOrigin.Begin);
            Assembly assembly = Assembly.Load(ms.ToArray());

            // 5. Retrieving the Test type
            Type? testType = assembly.GetType($"{namespaceName}.{className}");
            Assert.IsNotNull(testType, $"Type '{className}' not found in the namespace '{namespaceName}' in the compiled assembly.");

            Console.WriteLine("Compilation and Type Loading Successful!");
            foreach (var m in testType.GetMethods(BindingFlags.Public | BindingFlags.Static))
            {
                Console.WriteLine($"Found method: {m.Name}");
            }
        }
    }
}

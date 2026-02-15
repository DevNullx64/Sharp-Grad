using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.DifEngine.SyntaxBuilder;
using System.Linq;

namespace TestProject.Compilation
{
    [TestClass]
    public class ComputationGraphNodeExtensionsTests
    {
        [TestMethod]
        public void GetParallelSubgraphsDFS_WithRealNodes_Works()
        {
            Dimension Da = new(nameof(Da), 2);
            Dimension Db = new(nameof(Db), 2);

            // Variables
            Variable<float> a = new(nameof(a), [1, 2], Da);
            Variable<float> b = new(nameof(b), [3, 4], Db);

            // A+B
            BinaryComputedValue<float> a_plus_b = a + b;

            // sum(A) and sum(A+B) (in general, the barrier is set by the type or context)
            Value<float> sum_a = VMath.Sum(a, Da);
            Value<float> sum_a_plus_b = VMath.Sum(a_plus_b, Db);

            // Check that these are reduction barriers
            Assert.IsTrue(((Value)sum_a).Kind.IsReduction(), "sumA should be a reduction barrier.");
            Assert.IsTrue(((Value)sum_a_plus_b).Kind.IsReduction(), "sumAplusB should be a reduction barrier.");

            // sum(A) + sum(A+B)
            BinaryComputedValue<float> add_sums = sum_a + sum_a_plus_b;

            // root
            BinaryComputedValue<float> root = add_sums;

            // Call the method to test
            Value[][] subgraphs = root.GetParallelSubgraphsDFS(node => node.Kind.IsReduction());

            // There should be 3 subgraphs: [sumA], [sumAplusB], [addSums]
            Assert.AreEqual(
                3, subgraphs.Length,
                $"There should be 3 subgraphs ({nameof(sum_a)}, {nameof(sum_a_plus_b)}, {nameof(add_sums)}). Found: {subgraphs.Length}"
            );

            // Check the contents of each subgraph
            Assert.AreEqual(
                root, subgraphs.Last().Last(),
                $"The third subgraph should contain addSums as its last node."
            );
        }
    }
}
using SharpGrad;
using SharpGrad.DifEngine;
using SharpGrad.ExprLambda;
using SharpGrad.Operators;

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
            Variable<float> a = new([1, 2], Da, nameof(a));
            Variable<float> b = new([3, 4], Db, nameof(b));

            // A+B
            AddValue<float> a_plus_b = a + b;

            // sum(A) and sum(A+B) (in general, the barrier is set by the type or context)
            SumValue<float> sum_a = VMath.Sum(a, Da);
            SumValue<float> sum_a_plus_b = VMath.Sum(a_plus_b, Db);

            // Check that these are indeed parallel barriers
            Assert.IsTrue(sum_a.IsParallelBarrier, "sumA should be a parallel barrier.");
            Assert.IsTrue(sum_a_plus_b.IsParallelBarrier, "sumAplusB should be a parallel barrier.");

            // sum(A) + sum(A+B)
            AddValue<float> add_sums = sum_a + sum_a_plus_b;

            // root
            AddValue<float> root = add_sums;

            // Call the method to test
            Value[][] subgraphs = root.GetParallelSubgraphsDFS<Value>();

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
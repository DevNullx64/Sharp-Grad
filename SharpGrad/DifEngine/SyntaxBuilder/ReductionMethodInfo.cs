using System.Collections.Generic;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public readonly struct ReductionMethodInfo
    {
        public readonly string MethodSuffix;
        public readonly List<string> InputIndices;
        public readonly int ReduceDimension;

        private ReductionMethodInfo(string methodSuffix, List<string> inputIndices, int reduceDimension)
        {
            MethodSuffix = methodSuffix;
            InputIndices = inputIndices;
            ReduceDimension = reduceDimension;
        }

        public static ReductionMethodInfo Create(Shape input, int reduceDimension)
        {
            string methodSuffix = $"{input.Rank}_{reduceDimension}";
            List<string> inputIndices = new(input.Rank);
            for (int d = 0; d < input.Rank; d++)
            {
                string name = $"i{d}";
                inputIndices.Add(name);
            }
            return new(methodSuffix, inputIndices, reduceDimension);
        }
    }
}
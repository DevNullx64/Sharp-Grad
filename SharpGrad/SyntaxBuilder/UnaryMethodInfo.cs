using System.Collections.Generic;
using System.Text;

namespace SharpGrad.SyntaxBuilder
{
    public readonly struct UnaryMethodInfo
    {
        public readonly string MethodSuffix;
        public readonly List<string> InputIndices;

        private UnaryMethodInfo(string methodSuffix, List<string> inputIndices)
        {
            MethodSuffix = methodSuffix;
            InputIndices = inputIndices;
        }

        public static UnaryMethodInfo Create(Shape input)
        {
            StringBuilder sb = new();
            List<string> inputIndices = new(input.Rank);
            int rank = input.Rank;
            for (int d = 0; d < rank; d++)
            {
                sb.Append(d);
                string name = $"i{d}";
                inputIndices.Add(name);
            }
            string methodName = sb.ToString();
            return new UnaryMethodInfo(methodName, inputIndices);
        }
    }
}
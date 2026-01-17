using System.Collections.Generic;
using System.Text;

namespace SharpGrad.SyntaxBuilder
{
    public readonly struct BinaryMethodInfo
    {
        public readonly string MethodSuffix;
        public readonly List<string> LeftIndices;
        public readonly List<string> RightIndices;
        public readonly List<string> ResultIndices;

        private BinaryMethodInfo(string methodSuffix, List<string> leftIndices, List<string> rightIndices, List<string> resultIndices)
        {
            MethodSuffix = methodSuffix;
            LeftIndices = leftIndices;
            RightIndices = rightIndices;
            ResultIndices = resultIndices;
        }

        public static BinaryMethodInfo Create(Shape left, Shape right)
        {
            StringBuilder sb = new();
            List<string> resultIndices = [];
            List<string> leftIndices = new(left.Rank);
            int lRank = left.Rank;
            for (int d = 0; d < lRank; d++)
            {
                sb.Append(d);
                string name = $"il{d}";
                leftIndices.Add(name);
                resultIndices.Add(name);
            }

            sb.Append('_');

            char rightOnly = 'a';
            List<string> rightIndices = new(right.Rank);
            int rRank = right.Rank;
            for (int d = 0; d < rRank; d++)
            {
                int pos = left.IndexOf(right[d]);
                if (pos >= 0)
                {
                    sb.Append(pos);
                    rightIndices.Add(leftIndices[pos]);
                }
                else
                {
                    sb.Append(rightOnly++);
                    string name = $"ir{d}";
                    rightIndices.Add(name);
                    resultIndices.Add(name);
                }
            }
            string methodName = sb.ToString();
            return new BinaryMethodInfo(methodName, leftIndices, rightIndices, resultIndices);
        }
    }
}
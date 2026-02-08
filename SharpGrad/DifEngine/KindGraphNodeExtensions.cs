using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Extension methods for KindGraphNode enum.
    /// </summary>
    public static class KindGraphNodeExtensions
    {
        private static KindGraphNode[] GetAllKind()
        {
            KindGraphNode[] result = (KindGraphNode[])Enum.GetValues(typeof(KindGraphNode));
            Array.Sort(result);
            return result;
        }
        private static readonly KindGraphNode[] _allKinds = GetAllKind();

        public static bool IsValidKind(this KindGraphNode kind)
            => Array.BinarySearch(_allKinds, kind) >= 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsValue(this KindGraphNode kind)
            => kind == KindGraphNode.Constant || kind == KindGraphNode.Variable;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsFunction(this KindGraphNode kind)
            => kind == KindGraphNode.Function || kind == KindGraphNode.DifferentiableFunction;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsUnary(this KindGraphNode kind)
            => ((int)kind & (int)KindCategory.Mask) == (int)KindCategory.Unary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsBinary(this KindGraphNode kind)
            => ((int)kind & (int)KindCategory.Mask) == (int)KindCategory.Binary;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsReduction(this KindGraphNode kind)
            => ((int)kind & (int)KindCategory.Mask) == (int)KindCategory.Reduction;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsComparison(this KindGraphNode kind)
            => ((int)kind & (int)KindCategory.Mask) == (int)KindCategory.Comparison;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsCommutative(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.Commutative) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAssociative(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.Associative) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsDifferentiable(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.DifferentiabilityTypeMask) == (int)KindProperty.Differentiable;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsInverse(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.Inverse) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsConditionallyDifferentiable(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.DifferentiabilityTypeMask) == (int)KindProperty.ConditionallyDifferentiable;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPartiallyDifferentiable(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.DifferentiabilityTypeMask) == (int)KindProperty.PartiallyDifferentiable;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNotDifferentiable(this KindGraphNode kind)
            => ((int)kind & (int)KindProperty.DifferentiabilityTypeMask) == 0;


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static KindBinary GetBaseOperation(this KindReduction kindReduction)
        {
            KindGraphNode result = (KindGraphNode)((int)kindReduction & ~(int)KindCategory.Mask | (int)KindCategory.Binary);
            if (result.IsValidKind())
            {
                return (KindBinary)result;
            }
            throw new ArgumentException($"Reduction operation {kindReduction} does not have a corresponding base binary operation.");
        }

        public static KindGraphNode GetInverse(this KindGraphNode kind)
        {
            KindGraphNode inverted = kind ^ (KindGraphNode)KindProperty.Inverse;
            if (inverted.IsValidKind())
            {
                return inverted;
            }
            throw new ArgumentException($"Operation {kind} does not have an inverse.");
        }


        public static Shape GetResultShape(this KindGraphNode kind, Shape shape1, Shape? shape2 = null, Dimension[]? reduceDims = null)
        {
            if (kind.IsUnary())
            {
                if (shape2 is not null)
                {
                    throw new ArgumentException("Unary operations require only one shape.", nameof(shape2));
                }
                return ((KindUnary)kind).GetResultShape(shape1);
            }
            else if (kind.IsBinary())
            {
                if (shape2 == null)
                {
                    throw new ArgumentNullException(nameof(shape2), "Binary operations require two shapes.");
                }
                return ((KindBinary)kind).GetResultShape(shape1, shape2.Value);
            }
            else if (kind.IsReduction())
            {
                if (reduceDims == null || reduceDims.Length == 0)
                {
                    throw new ArgumentNullException(nameof(reduceDims), "Reduction operations require reduction dimensions.");
                }
                return ((KindReduction)kind).GetResultShape(shape1, reduceDims);
            }
            else
            {
                throw new NotImplementedException($"Result shape calculation not implemented for {nameof(KindGraphNode)}.{kind}.");
            }
        }
        public static Shape GetResultShape(this KindUnary kind, Shape shape1)
        {
            return shape1;
        }
        public static Shape GetResultShape(this KindBinary kind, Shape shape1, Shape shape2)
        {
            return Shape.Broadcast(shape1, shape2);
        }
        public static Shape GetResultShape(this KindReduction kind, Shape shape1, Dimension[] reduceDims)
        {
            return shape1.Remove(reduceDims);
        }


        public static string ToSymbol(this KindGraphNode kind)
        {
            return kind switch
            {
                // Value operations
                KindGraphNode.Constant => "Cx",
                KindGraphNode.Variable => "Vx",

                // Function operations
                KindGraphNode.DifferentiableFunction => "df(x)",
                KindGraphNode.Function => "f(x)",

                // Unary operations
                KindGraphNode.Negate => " -",
                KindGraphNode.Reciprocal => " 1/",
                KindGraphNode.Sqrt => " √",
                KindGraphNode.Sq => " ²",
                KindGraphNode.Exp => " exp",
                KindGraphNode.Log => " log",
                KindGraphNode.Log10 => " log10",
                KindGraphNode.Sin => " sin",
                KindGraphNode.Asin => " asin",
                KindGraphNode.Cos => " cos",
                KindGraphNode.Acos => " acos",
                KindGraphNode.Tan => " tan",
                KindGraphNode.Atan => " atan",
                KindGraphNode.Sinh => " sinh",
                KindGraphNode.Asinh => " asinh",
                KindGraphNode.Cosh => " cosh",
                KindGraphNode.Acosh => " acosh",
                KindGraphNode.Tanh => " tanh",
                KindGraphNode.Atanh => " atanh",
                KindGraphNode.Floor => " floor",
                KindGraphNode.Ceil => " ceil",
                KindGraphNode.Trunc => " trunc",
                KindGraphNode.Round => " round",
                KindGraphNode.Abs => " |x|",
                KindGraphNode.Sign => " sign",
                KindGraphNode.Cast => " (cast) ",

                // Binary operations
                KindGraphNode.Add => " + ",
                KindGraphNode.Subtract => " - ",
                KindGraphNode.Multiply => " * ",
                KindGraphNode.Divide => " / ",
                KindGraphNode.Power => " ^ ",
                KindGraphNode.Modulo => " % ",
                KindGraphNode.Min => " min",
                KindGraphNode.Max => " max",

                // Reduction operations
                KindGraphNode.Sum => " Σ",
                KindGraphNode.Product => " Π",
                // KindGraphNode.Mean => " mean",
                // KindGraphNode.MaxOf => " max",
                // KindGraphNode.MinOf => " min",

                // Comparison operations
                KindGraphNode.Equal => " == ",
                KindGraphNode.NotEqual => " != ",
                KindGraphNode.GreaterThan => " > ",
                KindGraphNode.GreaterEqual => " >= ",
                KindGraphNode.LessThan => " < ",
                KindGraphNode.LessEqual => " <= ",
                _ => throw new ArgumentOutOfRangeException(nameof(kind), $"No symbol defined for GraphNodeKind.{kind}"),
            };
        }


        public static T GetNeutralElement<T>(this KindGraphNode kind)
            where T : INumber<T>
        {
            return kind switch
            {
                KindGraphNode.Add => T.AdditiveIdentity,
                KindGraphNode.Subtract => T.AdditiveIdentity,
                KindGraphNode.Multiply => T.MultiplicativeIdentity,
                KindGraphNode.Divide => T.MultiplicativeIdentity,
                KindGraphNode.Min => T.CreateSaturating(double.PositiveInfinity),
                KindGraphNode.Max => T.CreateSaturating(double.NegativeInfinity),
                _ => throw new ArgumentException($"No neutral element defined for {nameof(KindGraphNode)}.{kind}", nameof(kind)),
            };
        }
    }
}

using System;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public static class TypeExtender
    {
        public static string GetFullName(this Type type, params string[] genericArgs)
        {
            return type.IsGenericType
                ? type.FullName!.Replace("`" + type.GetGenericArguments().Length, "") + "<" + string.Join(", ", genericArgs) + ">"
                : type.FullName!;
        }
        public static string GetName(this Type type, params string[] genericArgs)
        {
            return type.IsGenericType
                ? type.Name.Replace("`" + type.GetGenericArguments().Length, "") + "<" + string.Join(", ", genericArgs) + ">"
                : type.Name;
        }
    }
}
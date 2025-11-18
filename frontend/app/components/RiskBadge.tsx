import React from "react";
import { View, Text, StyleSheet } from "react-native";

export default function RiskBadge({ risk }) {
  const getRiskColor = () => {
    switch (risk?.toLowerCase()) {
      case "high":
        return "#FF6B6B";
      case "medium":
        return "#FFA500";
      case "low":
        return "#51CF66";
      default:
        return "#999";
    }
  };

  return (
    <View style={[styles.badge, { backgroundColor: getRiskColor() }]}>
      <Text style={styles.text}>{risk}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    alignSelf: "flex-start",
  },
  text: {
    color: "#fff",
    fontSize: 12,
    fontWeight: "bold",
  },
});

import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, Animated, FlatList, ScrollView } from "react-native";
import { colors } from "../theme/colors";
import AlertCard from "./components/AlertCard";
import SectionHeader from "./components/SectionHeader";

const mockCallAlerts = [
  {
    id: "1",
    resident: "Margaret Johnson",
    type: "Scammer claiming to be Tech Support",
    preview: "Caller claimed to be from Microsoft and asked for account access",
    risk: "High",
    timestamp: "5 min ago",
  },
  {
    id: "2",
    resident: "Robert Miller",
    type: "Suspicious IRS call",
    preview: "Caller threatened legal action if payment not made immediately",
    risk: "High",
    timestamp: "20 min ago",
  },
  {
    id: "3",
    resident: "Helen Davis",
    type: "Unknown caller",
    preview: "Caller refused to identify organization, requested bank details",
    risk: "Medium",
    timestamp: "45 min ago",
  },
];

export default function CallMonitoringScreen({ navigation }) {
  const [pulseAnim] = useState(new Animated.Value(1));
  const [isMonitoring] = useState(true);

  useEffect(() => {
    const startPulse = () => {
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.3,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ]).start(() => startPulse());
    };

    if (isMonitoring) {
      startPulse();
    }
  }, [isMonitoring, pulseAnim]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Penny Header */}
      <View style={styles.pennyHeader}>
        <View style={styles.pennyIcon}>
          <Text style={styles.pennyEmoji}>🐷</Text>
        </View>
        <Text style={styles.pennyName}>Penny</Text>
      </View>

      <SectionHeader
        title="Call Monitoring"
        subtitle="Real-time fraud detection"
      />

      <View style={styles.statusCard}>
        <Text style={styles.statusLabel}>MONITORING STATUS</Text>
        <View style={styles.statusContent}>
          <Text style={styles.statusText}>
            {isMonitoring ? "MONITORING: ON" : "MONITORING: OFF"}
          </Text>
          <Animated.View
            style={[
              styles.pulsingDot,
              { transform: [{ scale: pulseAnim }] },
            ]}
          />
        </View>
      </View>

      <View style={styles.activeCallCard}>
        <Text style={styles.activeCallLabel}>CURRENT CALL</Text>
        <Text style={styles.residentName}>Eleanor White</Text>
        <Text style={styles.callDuration}>Duration: 3 minutes 22 seconds</Text>
        <View style={styles.riskIndicator}>
          <View style={styles.riskDot} />
          <Text style={styles.riskText}>Analyzing for suspicious patterns...</Text>
        </View>
      </View>

      {/* Penny Message */}
      <View style={styles.pennyMessage}>
        <View style={styles.pennyMessageIcon}>
          <Text style={styles.pennyMessageEmoji}>🐷</Text>
        </View>
        <Text style={styles.pennyMessageText}>
          Penny quietly listens as your staff can focus on care.
        </Text>
      </View>

      <View style={styles.alertsSection}>
        <SectionHeader 
        title="Recent Call Alerts"
        subtitle="" />
        <FlatList
          data={mockCallAlerts}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <AlertCard
              item={item}
              onPress={() => navigation.navigate("AlertDetails", { alert: item })}
            />
          )}
          scrollEnabled={false}
        />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#FDF2F8",
  },
  contentContainer: {
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  pennyHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 20,
  },
  pennyIcon: {
    width: 48,
    height: 48,
    backgroundColor: "#FBCFE8",
    borderRadius: 24,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  pennyEmoji: {
    fontSize: 28,
  },
  pennyName: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#1F2937",
  },
  statusCard: {
    backgroundColor: "#FCE7F3",
    borderRadius: 20,
    padding: 24,
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  statusLabel: {
    fontSize: 11,
    color: "#9CA3AF",
    marginBottom: 12,
    fontWeight: "bold",
    letterSpacing: 1,
  },
  statusContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  statusText: {
    fontSize: 22,
    fontWeight: "bold",
    color: "#1F2937",
    letterSpacing: 0.5,
  },
  pulsingDot: {
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: "#10B981",
    shadowColor: "#10B981",
    shadowOpacity: 0.7,
    shadowRadius: 10,
    elevation: 5,
  },
  activeCallCard: {
    backgroundColor: "#EFF6FF",
    borderLeftWidth: 4,
    borderLeftColor: "#10B981",
    borderRadius: 20,
    padding: 20,
    marginBottom: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  activeCallLabel: {
    fontSize: 11,
    color: "#9CA3AF",
    marginBottom: 10,
    fontWeight: "bold",
    letterSpacing: 1,
  },
  residentName: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#1F2937",
    marginBottom: 4,
  },
  callDuration: {
    fontSize: 13,
    color: "#6B7280",
    marginBottom: 12,
  },
  riskIndicator: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  riskDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#10B981",
  },
  riskText: {
    fontSize: 13,
    color: "#10B981",
    fontWeight: "500",
  },
  pennyMessage: {
    flexDirection: "row",
    alignItems: "flex-start",
    backgroundColor: "#FCE7F3",
    borderRadius: 20,
    padding: 16,
    marginBottom: 20,
  },
  pennyMessageIcon: {
    width: 40,
    height: 40,
    backgroundColor: "#FBCFE8",
    borderRadius: 20,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  pennyMessageEmoji: {
    fontSize: 22,
  },
  pennyMessageText: {
    flex: 1,
    fontSize: 13,
    color: "#374151",
    lineHeight: 20,
    marginTop: 8,
  },
  alertsSection: {
    marginBottom: 20,
  },
});